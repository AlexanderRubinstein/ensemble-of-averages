import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from einops import rearrange


EPS = 1e-6


def budget_modifier(outputs, target):
    with torch.no_grad():
        ensemble_output = compute_ensemble_output(outputs)

        unreduced_ce = F.cross_entropy(
            ensemble_output,
            target,
            reduction='none'
        )
        divider = torch.pow(unreduced_ce.mean(0), 2)
        return unreduced_ce / divider


class A2DLoss(torch.nn.Module):
    def __init__(self, heads, dbat_loss_type='v1', reduction="mean"):
        super().__init__()
        self.heads = heads
        self.dbat_loss_type = dbat_loss_type
        self.reduction = reduction

    # input has shape [batch_size, heads * classes]
    def forward(self, logits):
        logits_chunked = torch.chunk(logits, self.heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=0).softmax(-1)
        m_idx = torch.randint(0, self.heads, (1,)).item()
        # shape [models, batch, classes]
        return a2d_loss_impl(
            probs,
            m_idx,
            dbat_loss_type=self.dbat_loss_type,
            reduction=self.reduction
        )


# when more than 2 models are used, pick random model and treat it as model with prob p2
# based on https://github.com/mpagli/Agree-to-Disagree/blob/d8859164025421e137dca8226ef3b10859bc276c/src/main.py#L92
def a2d_loss_impl(probs, m_idx, dbat_loss_type='v1', reduction='mean'):

    if dbat_loss_type == 'v1':
        adv_loss = []

        p_1_s, indices = [], []

        for i, p_1 in enumerate(probs):
            if i == m_idx:
                continue
            p_1, idx = p_1.max(dim=1)
            p_1_s.append(p_1)
            indices.append(idx)

        p_2 = probs[m_idx]

        # probs for classes predicted by each other model
        p_2_s = [p_2[torch.arange(len(p_2)), max_idx] for max_idx in indices]

        for i in range(len(p_1_s)):
            al = (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) + EPS))
            if reduction == 'mean':
                al = al.mean()
            else:
                assert reduction == 'none'

            adv_loss.append(al)
    # elif dbat_loss_type == 'v2':
    #     adv_loss = []
    #     p_2 = torch.softmax(m(x_tilde), dim=1)
    #     p_2_1, max_idx = p_2.max(dim=1) # proba of class 1 for m

    #     with torch.no_grad():
    #         p_1_s = [torch.softmax(m_(x_tilde), dim=1) for m_ in ensemble[:m_idx]]
    #         p_1_1_s = [p_1[torch.arange(len(p_1)), max_idx] for p_1 in p_1_s] # probas of class 1 for m_

    #     for i in range(len(p_1_s)):
    #         al = (- torch.log(p_1_1_s[i] * (1.0 - p_2_1) + p_2_1 * (1.0 - p_1_1_s[i]) +  1e-7)).mean()
    #         adv_loss.append(al)

    else:
        raise NotImplementedError("v2 dbat is not implemented yet")

    if reduction == "none":
        agg_func = func_for_dim(torch.mean, 0)
    else:
        assert reduction == "mean"
        agg_func = torch.mean
    return aggregate_tensors_by_func(adv_loss, func=agg_func)


# copied from: https://github.com/yoonholee/DivDis/blob/main/divdis.py#L6
def to_probs(logits, heads):
    """
    Converts logits to probabilities.
    Input must have shape [batch_size, heads * classes].
    Output will have shape [batch_size, heads, classes].
    """

    B, N = logits.shape
    if N == heads:  # Binary classification; each head outputs a single scalar.
        preds = logits.sigmoid().unsqueeze(-1)
        probs = torch.cat([preds, 1 - preds], dim=-1)
    else:
        logits_chunked = torch.chunk(logits, heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=1).softmax(-1)
    B, H, D = probs.shape
    assert H == heads
    return probs


# copied from: https://github.com/yoonholee/DivDis/blob/main/divdis.py#L46
class DivDisLoss(torch.nn.Module):
    """Computes pairwise repulsion losses for DivDis.

    Args:
        logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * DIM].
        heads (int): Number of heads.
        mode (str): DIVE loss mode. One of {pair_mi, total_correlation, pair_l1}.
    """

    def __init__(self, heads, mode="mi", reduction="mean"):
        super().__init__()
        self.heads = heads
        self.mode = mode
        self.reduction = reduction

    def forward(self, logits):
        heads, mode, reduction = self.heads, self.mode, self.reduction
        probs = to_probs(logits, heads)
        return divdis_loss_forward_impl(probs, mode=mode, reduction=reduction)

        # if mode == "mi":  # This was used in the paper
        #     marginal_p = probs.mean(dim=0)  # H, D
        #     marginal_p = torch.einsum(
        #         "hd,ge->hgde", marginal_p, marginal_p
        #     )  # H, H, D, D
        #     marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

        #     joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(
        #         dim=0
        #     )  # H, H, D, D
        #     joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

        #     # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
        #     # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
        #     kl_computed = joint_p * (joint_p.log() - marginal_p.log())
        #     kl_computed = kl_computed.sum(dim=-1)
        #     kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
        #     repulsion_grid = -kl_grid
        # elif mode == "l1":
        #     dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
        #     dists = dists.sum(dim=-1).mean(dim=0)
        #     repulsion_grid = dists
        # else:
        #     raise ValueError(f"{mode=} not implemented!")

        # if reduction == "mean":  # This was used in the paper
        #     repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
        #     repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
        #     repulsion_loss = -repulsions.mean()
        # elif reduction == "min_each":
        #     repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
        #         repulsion_grid, diagonal=-1
        #     )
        #     rows = [r for r in repulsion_grid]
        #     row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
        #     repulsion_loss = -torch.stack(row_mins).mean()
        # else:
        #     raise ValueError(f"{reduction=} not implemented!")

        # return repulsion_loss


def divdis_loss_forward_impl(probs, mode="mi", reduction="mean"):
    # input has shape [batch_size, heads, classes]
    # probs = to_probs(logits, heads)
    heads = probs.shape[1]

    if mode == "mi":  # This was used in the paper
        marginal_p = probs.mean(dim=0)  # H, D
        marginal_p = torch.einsum(
            "hd,ge->hgde", marginal_p, marginal_p
        )  # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

        joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(
            dim=0
        )  # H, H, D, D
        joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

        # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
        # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
        kl_computed = joint_p * (joint_p.log() - marginal_p.log() + EPS)
        kl_computed = kl_computed.sum(dim=-1)
        kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
        repulsion_grid = -kl_grid
    elif mode == "l1":
        dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
        dists = dists.sum(dim=-1).mean(dim=0)
        repulsion_grid = dists
    else:
        raise ValueError(f"{mode=} not implemented!")

    if reduction == "mean":  # This was used in the paper
        repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
        repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
        repulsion_loss = -repulsions.mean()
    elif reduction == "min_each":
        repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
            repulsion_grid, diagonal=-1
        )
        rows = [r for r in repulsion_grid]
        row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
        repulsion_loss = -torch.stack(row_mins).mean()
    else:
        raise ValueError(f"{reduction=} not implemented!")

    return repulsion_loss


def bootstrap_ensemble_outputs(outputs, assert_len=True):
    if_stores_input = stores_input(outputs)
    if assert_len:
        assert if_stores_input
    if if_stores_input:
        return [output[1] for output in outputs]
    else:
        return outputs


class DivDisLossWrapper(torch.nn.Module):

    def __init__(
        self,
        task_loss,
        weight,
        mode="mi",
        reduction="mean",
        mapper=None,
        loss_type="divdis",
        use_always_labeled=False,
        modifier=None,
        gamma=2.0,
        disagree_after_epoch=0,
        manual_lambda=1.0,
        disagree_below_threshold=None,
        reg_mode=None,
        reg_weight=None,
        convex_sum=False
    ):
        super().__init__()
        self.repulsion_loss = None
        self.mode = mode
        self.reduction = reduction
        self.task_loss = task_loss
        self.weight = weight
        self.loss_type = loss_type
        # if mapper == "mvh":
        #     self.mapper = make_mvh_mapper()
        # else:
        #     assert mapper is None, "Only mvh mapper is supported"
        #     self.mapper = None
    # if mapper == "mvh":
    #         self.mapper = make_mvh_mapper()
    #     else:
        assert mapper is None, "Only mvh mapper is supported"
        self.mapper = None
        self.log_this_batch = True
        self.use_always_labeled = use_always_labeled
        self.modifier = modifier
        self.gamma = gamma
        self.epoch = 0
        self.disagree_after_epoch = disagree_after_epoch
        self.manual_lambda = manual_lambda
        self.disagree_below_threshold = disagree_below_threshold
        self.reg_mode = reg_mode
        self.reg_weight = reg_weight
        self.convex_sum = convex_sum

        if self.reg_mode is not None:
            assert self.reg_weight is not None

    def increase_epoch(self):
        self.epoch += 1

    def compute_modifier(self, outputs, targets):

        # if self.modifier == "focal":
        #     # (1 - pt) ** 2
        #     return aggregate_tensors_by_func(
        #         [
        #             focal_modifier(output, targets, self.gamma)
        #                 for output in outputs
        #         ]
        #     )
        # elif self.modifier == "budget":
        #     return budget_modifier(outputs, targets)
        if self.modifier == "budget":
            return budget_modifier(outputs, targets)
        else:
            assert self.modifier is None
            return 1.0

    def forward(self, outputs, targets, unlabeled_outputs=None):

        def zero_if_none(value):
            return (
                value.item()
                    if value is not None
                    else 0
            )

        def get_repulsion_loss(outputs, unlabeled_outputs, targets_values):
            if unlabeled_outputs is None:
                assert not outputs[0][1].requires_grad, \
                    "No unlabeled batch was provided during training"
                repulsion_loss = None
                modifier = None
            else:
                n_heads = len(unlabeled_outputs)
                if self.repulsion_loss is None:

                    if self.loss_type == "divdis":
                        assert self.modifier != "budget", \
                            "Budget modifier is not supported for DivDisLoss"
                        self.repulsion_loss = DivDisLoss(
                            n_heads,
                            self.mode,
                            self.reduction
                        )
                    else:
                        assert self.loss_type == "a2d"
                        reduction = "mean"
                        if self.modifier == "budget":
                            reduction = "none"
                        self.repulsion_loss = A2DLoss(
                            n_heads,
                            reduction=reduction
                        )
                else:
                    self.repulsion_loss.heads = n_heads

                if self.use_always_labeled:
                    modifier = self.compute_modifier(
                        unlabeled_outputs,
                        targets_values
                    )
                else:
                    assert self.modifier is None
                    modifier = 1.0

                if self.mapper is not None:
                    unlabeled_outputs = [self.mapper(output) for output in unlabeled_outputs]

                # [batch, n * classes]
                unlabeled_outputs_cat = torch.cat(
                    unlabeled_outputs,
                    axis=-1
                )

                repulsion_loss = self.repulsion_loss(unlabeled_outputs_cat)
            return repulsion_loss, modifier, unlabeled_outputs

        if self.use_always_labeled:
            assert unlabeled_outputs is None
            unlabeled_outputs = outputs

        cur_n_heads = len(outputs)
        # if cur_n_heads > MAX_MODELS_WITHOUT_OOM:
        #     metrics_mappings = OOM_SAFE_METRICS
        # else:
        #     metrics_mappings = METRICS

        outputs = bootstrap_ensemble_outputs(outputs)
        targets_values = targets.max(-1).indices
        for output in outputs:
            assert not torch.isnan(output).any(), "NaNs in outputs"
        if unlabeled_outputs is not None:
            unlabeled_outputs = bootstrap_ensemble_outputs(unlabeled_outputs)

        repulsion_loss, modifier, reg_loss = None, None, None

        if self.weight > 0 and self.epoch + 1 > self.disagree_after_epoch:
            if self.disagree_below_threshold is not None:
                assert self.modifier is None, \
                    "Can't use modifier with disagree_below_threshold"
                assert self.weight < 1, \
                    "Can't have lambda == 1 with disagree_below_threshold"
                assert self.use_always_labeled

                masks = [
                    (
                            take_from_2d_tensor(
                                get_probs(output),
                                targets_values,
                                dim=-1
                            )
                        >
                            self.disagree_below_threshold
                    )
                        for output
                        in outputs
                ]
                # take samples which are low prob for all models
                mask = torch.stack(masks).min(0).values
                unlabeled_outputs = [
                    output[~mask, ...]
                        for output
                        in outputs
                ]
                outputs = [output[mask, ...] for output in outputs]
                targets = targets[mask, ...]
            if (
                    unlabeled_outputs is not None
                and
                    len(unlabeled_outputs) > 0
                and
                    len(unlabeled_outputs[0]) > 0
            ):
                repulsion_loss, modifier, unlabeled_outputs = get_repulsion_loss(
                    outputs,
                    unlabeled_outputs,
                    targets_values
                )

                # reg_loss = get_regularizer(
                #     self.reg_mode,
                #     outputs,
                #     unlabeled_outputs
                # )
                reg_loss = None

        if repulsion_loss is not None:
            assert not torch.isnan(repulsion_loss).any(), "NaNs in repulsion_loss"

        task_loss_value = torch.Tensor([0])[0]
        total_loss = task_loss_value.to(targets.device)
        if self.weight < 1:
            if self.convex_sum:
                task_loss_value = aggregate_tensors_by_func(
                    [self.task_loss(output, targets) for output in outputs],
                    func=func_for_dim(torch.mean, 0)
                )
                total_loss = task_loss_value
            else:

                if len(outputs) > 0 and len(outputs[0]) > 0:
                    task_loss_value = aggregate_tensors_by_func(
                        [self.task_loss(output, targets) for output in outputs]
                    )

                    total_loss = (1 - self.weight) * task_loss_value
        else:
            assert self.weight == 1
            assert self.disagree_after_epoch == 0, \
                "When lambda is 1 disagreement should start from the first epoch"

        if repulsion_loss is not None:

            if self.convex_sum:

                div_weight = modifier * self.weight
                assert (div_weight <= 1).all(), \
                    "Total diversity weight should be <= 1 when convex_sum is True"

                total_loss = (1 - div_weight) * total_loss + self.manual_lambda * div_weight * repulsion_loss
                total_loss = total_loss.mean()
                repulsion_loss = repulsion_loss.mean()
            else:
                repulsion_loss *= modifier

                if len(repulsion_loss.shape) > 0 and repulsion_loss.shape[0] > 1:
                    repulsion_loss = repulsion_loss.mean()


                total_loss += self.manual_lambda * self.weight * repulsion_loss

        if reg_loss is not None:
            total_loss += self.reg_weight * self.weight * reg_loss

        loss_info = {
            "task_loss": task_loss_value.item(),
            "repulsion_loss": zero_if_none(repulsion_loss),
            "regularizer_loss": zero_if_none(reg_loss),
            "total_loss": total_loss.item()
        }

        # if modifier is not None:
        #     loss_info["modifier_max_loss"] = modifier.max().item()
        #     loss_info["modifier_mean_loss"] = modifier.mean().item()
        #     loss_info["modifier_min_loss"] = modifier.min().item()

        # if self.log_this_batch:
        #     record_diversity(
        #         loss_info,
        #         outputs,
        #         torch.stack(outputs),
        #         metrics_mappings=metrics_mappings,
        #         name_prefix="ID_loss_"
        #     )
        #     if unlabeled_outputs is not None and not self.use_always_labeled:

        #         record_diversity(
        #             loss_info,
        #             unlabeled_outputs,
        #             torch.stack(unlabeled_outputs),
        #             metrics_mappings=metrics_mappings,
        #             name_prefix="OOD_loss_"
        #         )
        gradients_info = {}
        return total_loss, loss_info, gradients_info


def stores_input(outputs):
    assert len(outputs) > 0
    output_0 = outputs[0]
    return isinstance(output_0, (list, tuple)) and len(output_0) == 2


def compute_ensemble_output(
    outputs,
    weights=None,
    process_logits=None
):

    if process_logits is None:
        process_logits = lambda x: x

    if weights is None:
        weights = [1.0] * len(outputs)

    if stores_input(outputs):
        extractor = lambda x: x[1]
    else:
        extractor = lambda x: x

    return aggregate_tensors_by_func(
        [
            weight * process_logits(extractor(submodel_output).unsqueeze(0))
                for weight, submodel_output
                    in zip(weights, outputs)
        ],
        func=func_for_dim(torch.mean, dim=0)
    ).squeeze(0)


def normalize_weights(weights, p=2):
    assert len(weights.shape) == 1
    if not torch.isclose(torch.linalg.norm(weights, p), torch.Tensor([1])).all():
        weights = torch.nn.functional.normalize(weights, p=p, dim=0)
    return weights


def aggregate_tensors_by_func(input_list, func=torch.mean):
    return func(
        torch.stack(
            input_list
        )
    )


def func_for_dim(func, dim):

    def inner_func(*args, **kwargs):
        return func(*args, **kwargs, dim=dim)

    return inner_func


def are_probs(logits):
    if (
            logits.min() >= 0
        and
            logits.max() <= 1
        # don't check sums to one for the cases
        # like IN_A where masking drops some probs

        and
            abs(logits.sum(-1)[0][0] - 1) > EPS
    ):
        return True
    return False


def get_probs(logits):
    if are_probs(logits):
        probs = logits
    else:
        probs = F.softmax(logits, dim=-1)
    return probs


class RedneckEnsemble(nn.Module):
    def __init__(
        self,
        # n_models,
        # base_model_builder,
        models_list,
        weights=None,
        single_model_per_epoch=False,
        identical=False,
        feature_extractor=None,
        product_of_experts=False,
        random_select=None,
        keep_inactive_on_cpu=False,
        softmax_ensemble=False,
        split_last_linear_layer=False,
        freeze_feature_extractor=True
    ):
        # def one_layer_model(submodels):
        #     for submodel in submodels:
        #         if not is_mlp(submodel):
        #             return False
        #         # expect only linear + dropout
        #         if len(submodel.mlp._modules) > 2:
        #             return False
        #     return True

        # assert n_models > 0, "Need to have at least one model in ensemble"
        # assert not (single_model_per_epoch and product_of_experts), \
        #     f"Cannot use {SINGLE_MODEL_KEY} and {POE_KEY} together"
        # if single_model_per_epoch or product_of_experts:
        #     assert weights is None, \
        #         f"Cannot use weights with {SINGLE_MODEL_KEY} or {POE_KEY}"

        # if single_model_per_epoch:
        #     self.single_model_id = 0
        # else:
        #     self.single_model_id = None
        self.single_model_id = None
        # assert isinstance(base_model_builder, ModelBuilderBase)

        super(RedneckEnsemble, self).__init__()
        # self.n_models = n_models
        self.n_models = len(models_list)
        self.split_last_linear_layer = split_last_linear_layer
        # if identical:
        #     models_list = [base_model_builder.build()]
        #     for _ in range(self.n_models - 1):
        #         models_list.append(copy.deepcopy(models_list[0]))
        # else:
        #     if self.split_last_linear_layer:
        #         linear_layer = base_model_builder.build()
        #         assert isinstance(linear_layer, nn.Linear)
        #         models_list = split_linear_layer(linear_layer, self.n_models)
        #     else:
        #         models_list = [
        #             base_model_builder.build() for _ in range(self.n_models)
        #         ]
        self.submodels = nn.ModuleList(
            models_list
        )
        self.set_weights(weights)

        self.freeze_feature_extractor = freeze_feature_extractor

        self.feature_extractor = feature_extractor
        # if self.feature_extractor is not None and self.freeze_feature_extractor:
        #     unrequire_grads(self.feature_extractor, None)

        # self.product_of_experts = product_of_experts
        self.softmax = torch.nn.Softmax(dim=-1)

        self.random_select = random_select
        if self.random_select is not None:
            assert self.random_select > 1 and self.random_select <= self.n_models

        # self.one_layer_models = one_layer_model(self.submodels)
        self.latest_device = torch.device("cpu")
        self.keep_inactive_on_cpu = keep_inactive_on_cpu
        self.different_devices = False

        self.active_indices = None
        self.keep_active_indices = False

        self.soup = None
        self.softmax_ensemble = softmax_ensemble

    def apply_feature_extractor(self, x):
        if (
                hasattr(self, "feature_extractor")
            and
                self.feature_extractor is not None
        ):
            x = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        return x

    # def cook_soup(self):
    #     assert len(self.submodels) > 0
    #     base_model = self.submodels[0]
        # if isinstance(base_model, ModelClassesWrapper):
        #     base_model = base_model.get_inner_object()
    #     self.soup = copy.deepcopy(base_model)
    #     self.soup.load_state_dict(make_uniform_soup(self.submodels))

    def set_weights(self, weights, normalize=True):
        self.weights = weights

        if self.weights is not None and normalize:
            self.weights = normalize_weights(torch.Tensor(self.weights))

    def set_single_model_id(self, id):
        assert id >= 0 and id < self.n_models, \
            f"Invalid model id {id}, have only {self.n_models} models"
        self.single_model_id = id

    # def after_epoch(self, is_train, logger):
    #     if is_train and self.single_model_id is not None:
    #         new_id = cycle_id(self.single_model_id, self.n_models)
    #         logger.log(f"Switching single model id to {new_id}")
    #         self.set_single_model_id(new_id)

    def to(self, *args, **kwargs):
        # if isinstance(args[0], (str, torch.device)):
        #     self.latest_device = args[0]
        self.latest_device = args[0]

        # TODO(Alex | 24.01.2024): make sure that
        # torch.nn.module recursively calls this automatically
        # once ModuleDelegatingWrapper is fixed
        # in case submodels are wrapped
        for submodel in self.submodels:
            submodel.to(*args, **kwargs)
        if hasattr(self, "soup") and self.soup is not None:
            self.soup.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def forward(self, input):

        def equivalent_linear(active_submodels):
            cat_weight = []
            cat_bias = []
            for submodel in active_submodels:
                assert len(submodel.mlp._modules) == 2, \
                    "Expect only linear + dropout"
                linear_layer = submodel.mlp._modules["0"]
                cat_weight.append(linear_layer.weight)
                cat_bias.append(linear_layer.bias)

            cat_weight = torch.cat(cat_weight, dim=0)
            cat_bias = torch.cat(cat_bias, dim=0)
            cat_output = torch.nn.functional.linear(input, cat_weight, cat_bias)
            outputs = torch.chunk(cat_output, len(active_submodels), dim=-1)
            outputs = [[input, submodel_output] for submodel_output in outputs]
            return outputs

        def apply_submodel(submodel, input):
            if isinstance(submodel, torch.nn.Linear):
                input = input.squeeze(-1).squeeze(-1)
            return submodel(input)

        def select_active_submodels(
            submodels,
            random_select,
            training,
            keep_inactive_on_cpu,
            latest_device,
            different_devices,
            keep_active_indices,
            active_indices
        ):
            if not keep_active_indices:
                active_indices = None
            if training and random_select is not None:
                if active_indices is None:
                    active_indices = set(random.sample(
                        range(len(submodels)),
                        random_select
                    ))

                active_submodels = []
                for i in range(len(submodels)):
                    if i in active_indices:
                        active_submodels.append(submodels[i].to(latest_device))
                    elif keep_inactive_on_cpu:
                        different_devices = True
                        submodels[i].to("cpu")
            else:
                if different_devices:
                    for i in range(len(submodels)):
                        submodels[i].to(latest_device)
                    different_devices = False
                active_submodels = submodels
            return active_submodels, different_devices, active_indices

        if self.feature_extractor is not None:
            input = self.apply_feature_extractor(input)

        if self.single_model_id is not None and self.training:
            assert self.product_of_experts is False
            return self.submodels[self.single_model_id](input)

        # if self.product_of_experts and self.training:
        #     assert hasattr(self, "softmax")
        #     return aggregate_tensors_by_func(
        #         [
        #             torch.log(self.softmax(submodel(input)))
        #                 for submodel in self.submodels
        #         ],
        #         func=func_for_dim(torch.sum, dim=0)
        #     )

        (
            active_submodels,
            self.different_devices,
            self.active_indices
        ) = select_active_submodels(
            self.submodels,
            self.random_select,
            self.training,
            self.keep_inactive_on_cpu,
            self.latest_device,
            self.different_devices,
            self.keep_active_indices,
            self.active_indices
        )

        # if self.training and self.one_layer_models:
        #     outputs = equivalent_linear(active_submodels)
        # else:
        #     outputs = [
        #         [input, apply_submodel(submodel, input)]
        #             for submodel
        #                 in active_submodels
        #     ]
        outputs = [
            [input, apply_submodel(submodel, input)]
                for submodel
                    in active_submodels
        ]
        if self.weights is not None:
            if self.softmax_ensemble:
                process_logits = get_probs
            else:
                process_logits = None
            assert len(self.weights) == len(outputs)

            aggregated_output = compute_ensemble_output(
                outputs,
                self.weights,
                process_logits=process_logits
            )
            outputs.append([input, aggregated_output])
        if self.n_models == 1:
            assert len(outputs) == 1, \
                "n_models is 1, but number of outputs is not equal to 1, " \
                "maybe ensemble mode is on"

            assert len(outputs[0]) == 2
            outputs = outputs[0][1]
        return outputs

