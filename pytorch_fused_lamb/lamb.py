# adapted from: https://github.com/pytorch/pytorch/blob/1346ebf12e4c1a985ace72b187edaef4eae8b075/torch/optim/adam.py
from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, ParamsT, _use_grad_for_differentiable, _get_value,
                        _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach,
                        _get_scalar_dtype, _capturable_doc, _differentiable_doc, _foreach_doc,
                        _fused_doc, _maximize_doc, _view_as_real)
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

__all__ = ['Lamb', 'lamb']


class Lamb(Optimizer):
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.,
                 amsgrad: bool = False,
                 *,
                 maximize: bool = False,
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=True, capturable=True,
                        differentiable=False, fused=False)
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state['step']):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype(is_fused=False), device=p.device)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((), dtype=_get_scalar_dtype(is_fused=False), device=p.device)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            lamb(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
            )

        return loss


# TODO
Lamb.__doc__ = r"""Implements LAMB optimizer."""


def lamb(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         has_complex: bool = False,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: Union[float, Tensor],
         weight_decay: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs LAMB algorithm computation.

    See :class:`~pytorch_fused_lamb.Lamb` for details.
    """

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    func = _multi_tensor_lamb

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         has_complex=has_complex,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
       )


def _multi_tensor_lamb(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       state_steps: List[Tensor],
                       *,
                       amsgrad: bool,
                       has_complex: bool,
                       beta1: float,
                       beta2: float,
                       lr: Union[float, Tensor],
                       weight_decay: float,
                       eps: float,
                       maximize: bool,
                       ):
    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_state_steps,
    ), _) in grouped_tensors.values():

        # Handle complex parameters
        if has_complex:
            if amsgrad:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
        bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
#        # we do not negate bias_correction1 as it'll need to be negated later anyway
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)

        torch._foreach_sqrt_(bias_correction2)

        # Re-assign for clarity as we maintain minimal intermediates: we'll have
        # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
        bias_correction2_sqrt = bias_correction2

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

            # Set intermediate to the max. for normalizing running avg. of gradient when amsgrad
            exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

        # LAMB
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)
        # exp_avg_sq_sqrt is now bias_correction1 * sqrt(device_exp_avg_sqs / bias_correction2).
        # Next steps compute:
        # update = (device_exp_avgs / bias_correction1) / sqrt(device_exp_avg_sqs / bias_correction2)
        # while reusing the memory of exp_avg_sq_sqrt.
        torch._foreach_mul_(exp_avg_sq_sqrt, bias_correction1)
        torch._foreach_reciprocal_(exp_avg_sq_sqrt)
        torch._foreach_mul_(exp_avg_sq_sqrt, device_exp_avgs)
        update = exp_avg_sq_sqrt

        torch._foreach_add_(update, device_params, alpha=weight_decay)

        # Important part of LAMB: calculate ratio of param norm and update norm
        # NOTE: uses ord=2 like the official TF implementation
        update_norm = torch._foreach_norm(update, ord=2)
        p_norm = torch._foreach_norm(device_params, ord=2)
        torch._foreach_div_(p_norm, update_norm)
        trust_ratio = p_norm
        # TODO: if the update or parameter norm is 0, the trust ratio is defined as 1 (i.e., a normal adam update).
        # This is implemented below by substituting all inf ratios with 1 using succesive torch.where ops.
        # This needs to be changed to torch._foreach_where once it is available (https://github.com/pytorch/pytorch/issues/117884).
        # (Or, alternatively, to a _foreach version of torch.nan_to_num.)
        # Currently, this is the biggest performance bottleneck.
        trust_ratio = tuple(torch.where(torch.isinf(ratio), 1., ratio) for ratio in trust_ratio)
        

        step_size = trust_ratio
        torch._foreach_mul_(step_size, -lr)

        torch._foreach_mul_(update, step_size)
        torch._foreach_add_(device_params, update)

