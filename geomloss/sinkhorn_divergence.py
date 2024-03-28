"""Implements the (unbiased) Sinkhorn divergence between abstract measures.

.. math::
    \\text{S}_{\\varepsilon,\\rho}(\\alpha,\\beta) 
        ~&=~ \\text{OT}_{\\varepsilon,\\rho}(\\alpha, \\beta)
         ~-~\\tfrac{1}{2} \\text{OT}_{\\varepsilon,\\rho}(\\alpha, \\alpha)
         ~-~\\tfrac{1}{2} \\text{OT}_{\\varepsilon,\\rho}(\\beta, \\beta)
         ~+~ \\tfrac{\\varepsilon}{2} \| \\langle \\alpha, 1\\rangle - \\langle \\beta, 1\\rangle \|^2

where:

.. math::
    \\text{OT}_{\\varepsilon,\\rho}(\\alpha, \\beta)
    ~&=~ \\min_{\pi\geqslant 0} \\langle\, \pi\,,\, \\text{C} \,\\rangle
        ~+~\\varepsilon \, \\text{KL}(\pi,\\alpha\otimes\\beta) \\\\
        ~&+~\\rho \, \\text{KL}(\pi\,\mathbf{1},\\alpha)
        ~+~\\rho \, \\text{KL}(\pi^\intercal \,\mathbf{1},\\beta ) \\
    &=~ \\max_{b,a} -\\rho \\langle\, \\alpha \,,\, e^{-b/\\rho} - 1\,\\rangle
        -\\rho \\langle\, \\beta \,,\, e^{-a/\\rho} - 1\,\\rangle \\\\
        &-~
        \\epsilon \\langle\, \\alpha\otimes\\beta \,,\, e^{(b\oplus a - \\text{C})/\\epsilon} - 1\,\\rangle,

with a Kullback-Leibler divergence defined through:

.. math::
    \\text{KL}(\\alpha, \\beta)~=~
    \\langle \, \\alpha  \,,\, \\log \\tfrac{\\text{d}\\alpha}{\\text{d}\\beta} \,\\rangle
    ~-~ \\langle \, \\alpha  \,,\, 1 \,\\rangle
    ~+~ \\langle \, \\beta   \,,\, 1 \,\\rangle ~\geqslant~ 0.
"""

import numpy as np
import torch
from functools import partial
torch.cuda.set_device(0)

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch.cluster import (
        grid_cluster,
        cluster_ranges_centroids,
        sort_clusters,
        from_matrix,
    )

    keops_available = True
except:
    keops_available = False

from .utils import scal, squared_distances, distances


# ==============================================================================
#                            ε-scaling heuristic
# ==============================================================================


def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter


def epsilon_schedule(p, diameter, blur, scaling):
    ε_s = (
        [diameter ** p]
        + [
            np.exp(e)
            for e in np.arange(
                p * np.log(diameter), p * np.log(blur), p * np.log(scaling)
            )
        ]
        + [blur ** p]
    )
    return ε_s


def scaling_parameters(x, y, p, blur, reach, diameter, scaling):

    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    ε = blur ** p
    ε_s = epsilon_schedule(p, diameter, blur, scaling)
    ρ = None if reach is None else reach ** p
    return diameter, ε, ε_s, ρ


# ==============================================================================
#                              Sinkhorn loop
# ==============================================================================


def dampening(ε, ρ):
    return 1 if ρ is None else 1 / (1 + ε / ρ)


def log_weights(α):
    α_log = α.log()
    α_log[α <= 0] = -100000
    return α_log


class UnbalancedWeight(torch.nn.Module):
    def __init__(self, ε, ρ):
        super(UnbalancedWeight, self).__init__()
        self.ε, self.ρ = ε, ρ

    def forward(self, x):
        return (self.ρ + self.ε / 2) * x

    def backward(self, g):
        return (self.ρ + self.ε) * g


def sinkhorn_cost(
    ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=False, debias=True, potentials=False
):
    if potentials:  # Just return the dual potentials
        if debias:
            return b_x - a_x, a_y - b_y
        else:
            return b_x, a_y


    else:  # Actually compute the Sinkhorn divergence
        if (
            debias
        ):  # UNBIASED Sinkhorn divergence, S_ε(α,β) = OT_ε(α,β) - .5*OT_ε(α,α) - .5*OT_ε(β,β)
            if ρ is None:
                return scal(α, b_x - a_x, batch=batch) + scal(β, a_y - b_y, batch=batch)
            else:
                return scal(
                    α,
                    UnbalancedWeight(ε, ρ)((-a_x / ρ).exp() - (-b_x / ρ).exp()),
                    batch=batch,
                ) + scal(
                    β,
                    UnbalancedWeight(ε, ρ)((-b_y / ρ).exp() - (-a_y / ρ).exp()),
                    batch=batch,
                )

        else:  # Classic, BIASED entropized Optimal Transport OT_ε(α,β)
            if ρ is None:
                return scal(α, b_x, batch=batch) + scal(β, a_y, batch=batch)
            else:
                return scal(
                    α, UnbalancedWeight(ε, ρ)(1 - (-b_x / ρ).exp()), batch=batch
                ) + scal(β, UnbalancedWeight(ε, ρ)(1 - (-a_y / ρ).exp()), batch=batch)


def sinkhorn_loop(
    softmin,
    α_logs,
    β_logs,
    C_xxs,
    C_yys,
    C_xys,
    C_yxs,
    ε_s,
    ρ,
    jumps=[],
    kernel_truncation=None,
    truncate=5,
    cost=None,
    extrapolate=None,
    debias=True,
    last_extrapolation=True,
):

    Nits = len(ε_s)
    if type(α_logs) is not list:
        α_logs, β_logs = [α_logs], [β_logs]
        if debias:
            C_xxs, C_yys = [C_xxs], [C_yys]
        C_xys, C_yxs = [C_xys], [C_yxs]

    torch.autograd.set_grad_enabled(True)

    k = 0  # Scale index; we start at the coarsest resolution available
    ε = ε_s[k]
    λ = dampening(ε, ρ)
    #print(jumps)
    #print(ε_s)

    # Load the measures and cost matrices at the current scale:
    α_log, β_log = α_logs[k], β_logs[k]
    if debias:
        C_xx, C_yy = C_xxs[k], C_yys[k]
    C_xy, C_yx = C_xys[k], C_yxs[k]

    # Start with a decent initialization for the dual vectors:
    if debias:
        a_x = λ * softmin(ε, C_xx, α_log)  # OT(α,α)
        b_y = λ * softmin(ε, C_yy, β_log)  # OT(β,β)
    a_y = λ * softmin(ε, C_yx, α_log)  # OT(α,β) wrt. a
    b_x = λ * softmin(ε, C_xy, β_log)  # OT(α,β) wrt. b

    for i, ε in enumerate(ε_s):  # ε-scaling descent -----------------------

        λ = dampening(ε, ρ)  # ε has changed, so we should update λ too!

        # "Coordinate ascent" on the dual problems:
        if debias:
            at_x = λ * softmin(ε, C_xx, α_log + a_x / ε)  # OT(α,α)
            bt_y = λ * softmin(ε, C_yy, β_log + b_y / ε)  # OT(β,β)
        at_y = λ * softmin(ε, C_yx, α_log + b_x / ε)  # OT(α,β) wrt. a
        bt_x = λ * softmin(ε, C_xy, β_log + a_y / ε)  # OT(α,β) wrt. b

        # Symmetrized updates:
        if debias:
            a_x, b_y = 0.5 * (a_x + at_x), 0.5 * (b_y + bt_y)  # OT(α,α), OT(β,β)
        a_y, b_x = 0.5 * (a_y + at_y), 0.5 * (b_x + bt_x)  # OT(α,β) wrt. a, b

        if i in jumps:  # Jump from a coarse to a finer scale --------------

            if i == len(ε_s) - 1:  # Last iteration: just extrapolate!

                if debias:
                    C_xx_, C_yy_ = C_xxs[k + 1], C_yys[k + 1]
                C_xy_, C_yx_ = C_xys[k + 1], C_yxs[k + 1]

                last_extrapolation = False  # No need to re-extrapolate after the loop
                torch.autograd.set_grad_enabled(True)

            else:  # It's worth investing some time on kernel truncation...

                # Kernel truncation trick (described in Bernhard Schmitzer's 2016 paper),
                # that typically relies on KeOps' block-sparse routines:
                if debias:
                    C_xx_, _ = kernel_truncation(
                        C_xx,
                        C_xx,
                        C_xxs[k + 1],
                        C_xxs[k + 1],
                        a_x,
                        a_x,
                        ε,
                        truncate=truncate,
                        cost=cost,
                    )
                    C_yy_, _ = kernel_truncation(
                        C_yy,
                        C_yy,
                        C_yys[k + 1],
                        C_yys[k + 1],
                        b_y,
                        b_y,
                        ε,
                        truncate=truncate,
                        cost=cost,
                    )
                C_xy_, C_yx_ = kernel_truncation(
                    C_xy,
                    C_yx,
                    C_xys[k + 1],
                    C_yxs[k + 1],
                    b_x,
                    a_y,
                    ε,
                    truncate=truncate,
                    cost=cost,
                )

            # Extrapolation for the symmetric problems:
            if debias:
                a_x = extrapolate(a_x, a_x, ε, λ, C_xx, α_log, C_xx_)
                b_y = extrapolate(b_y, b_y, ε, λ, C_yy, β_log, C_yy_)

            # The cross-updates should be done in parallel!
            a_y, b_x = extrapolate(a_y, b_x, ε, λ, C_yx, α_log, C_yx_), extrapolate(
                b_x, a_y, ε, λ, C_xy, β_log, C_xy_
            )

            # Update the measure weights and cost "matrices":
            k = k + 1
            α_log, β_log = α_logs[k], β_logs[k]
            if debias:
                C_xx, C_yy = C_xx_, C_yy_
            C_xy, C_yx = C_xy_, C_yx_

    torch.autograd.set_grad_enabled(True)

    if last_extrapolation:
        # Last extrapolation, to get the correct gradients:
        if debias:
            a_x = λ * softmin(ε, C_xx, (α_log + a_x / ε).detach())
            b_y = λ * softmin(ε, C_yy, (β_log + b_y / ε).detach())

        # The cross-updates should be done in parallel!
        a_y, b_x = λ * softmin(ε, C_yx, (α_log + b_x / ε)), λ * softmin(
            ε, C_xy, (β_log + a_y / ε)
        )

    if debias:
        return a_x, b_y, a_y, b_x
    else:
        return None, None, a_y, b_x


def barycenter_loop(
    softmin,
    α_logs,
    β_logs,
    C_xzs,
    C_yzs,
    C_zxs,
    C_zys,
    ε_s,
    rho,
    lambda_=0.5,
    jumps=[],
    kernel_truncation=None,
    truncate=5,
    cost=None,
    extrapolate=None,
    debias=True,
    last_extrapolation=True,
):

    Nits = len(ε_s)
    if type(α_logs) is not list:
        α_logs, β_logs = [α_logs], [β_logs]
        C_xzs, C_yzs = [C_xzs], [C_yzs]
        C_zxs, C_zys = [C_zxs], [C_zys]

    #torch.autograd.set_grad_enabled(False)

    k = 0  # Scale index; we start at the coarsest resolution available
    ε = ε_s[k]
    λ = dampening(ε, rho)

    # Load the measures and cost matrices at the current scale:
    α_log, β_log = α_logs[k], β_logs[k]
    C_xz, C_yz = C_xzs[k], C_yzs[k]
    C_zx, C_zy = C_zxs[k], C_zys[k]

    # Initialize barycenter
    gamma_log = (lambda_ * (α_log.detach().exp()) +(1-lambda_)*(β_log.detach().exp())).log()
    d_log = torch.zeros_like(gamma_log)

    # Start with a decent initialization for the dual vectors:
    #u_1 = λ * softmin(ε, C_xz, gamma_log)  # OT(α,α)
    #u_2 = λ * softmin(ε, C_yz, gamma_log)  # OT(β,β)
    #v_1 = λ * softmin(ε, C_zx, β_log)  # OT(α,β) wrt. b
    #v_2 = λ * softmin(ε, C_zy, α_log)  # OT(α,β) wrt. a
    u_1 = torch.zeros_like(α_log)  # OT(α,α)
    u_2 = torch.zeros_like(β_log)  # OT(β,β)
    v_1 = torch.zeros_like(gamma_log)  # OT(α,β) wrt. b
    v_2 = torch.zeros_like(gamma_log) # OT(α,β) wrt. a

    for i, ε in enumerate(ε_s):  # ε-scaling descent -----------------------

        λ = dampening(ε, rho)  # ε has changed, so we should update λ too!
        for j in np.arange(2):
            # "Coordinate ascent" on the dual problems:
            u_1 = λ * softmin(ε, C_xz, gamma_log + v_1 / ε)  # OT(α,α)
            u_2 = λ * softmin(ε, C_yz, gamma_log + v_2 / ε)  # OT(β,β)
            gamma_log = lambda_*(-softmin(ε, C_zy, α_log + u_1 / ε)/ε)+(1-lambda_)* (- softmin(ε, C_zx, β_log + u_2 / ε)/ε) + d_log
            v_1 = λ * softmin(ε, C_zy, α_log + u_1 / ε)  # OT(α,β) wrt. a
            v_2 = λ * softmin(ε, C_zx, β_log + u_2 / ε)  # OT(α,β) wrt. b
            d_log = 0.5 * (d_log + gamma_log + softmin(ε, C_xz, d_log) / ε)

            # Symmetrized updates:
            #u_1, u_2 = 0.5 * (u_1 + u_1t), 0.5 * (u_2 + u_2t)  # OT(α,α), OT(β,β)
            #v_1, v_2 = 0.5 * (v_1 + v_1t), 0.5 * (v_2 + v_2t)  # OT(α,β) wrt. a, b

        if i in jumps:  # Jump from a coarse to a finer scale --------------

            if i == len(ε_s) - 1:  # Last iteration: just extrapolate!

                if debias:
                    C_xz_, C_yz_ = C_xzs[k + 1], C_yzs[k + 1]
                C_zx_, C_zy_ = C_zxs[k + 1], C_zys[k + 1]

                last_extrapolation = False  # No need to re-extrapolate after the loop
                torch.autograd.set_grad_enabled(True)

            else:  # It's worth investing some time on kernel truncation...

                # Kernel truncation trick (described in Bernhard Schmitzer's 2016 paper),
                # that typically relies on KeOps' block-sparse routines:
                if debias:
                    C_xz_, _ = kernel_truncation(
                        C_xz,
                        C_xz,
                        C_xzs[k + 1],
                        C_xzs[k + 1],
                        a_x,
                        a_x,
                        ε,
                        truncate=truncate,
                        cost=cost,
                    )
                    C_yz_, _ = kernel_truncation(
                        C_yz,
                        C_yz,
                        C_yzs[k + 1],
                        C_yzs[k + 1],
                        b_y,
                        b_y,
                        ε,
                        truncate=truncate,
                        cost=cost,
                    )
                C_zx_, C_zy_ = kernel_truncation(
                    C_zx,
                    C_zy,
                    C_zxs[k + 1],
                    C_zys[k + 1],
                    b_x,
                    a_y,
                    ε,
                    truncate=truncate,
                    cost=cost,
                )

            # Extrapolation for the symmetric problems:
            if debias:
                a_x = extrapolate(a_x, a_x, ε, λ, C_xz, α_log, C_xz_)
                b_y = extrapolate(b_y, b_y, ε, λ, C_yz, β_log, C_yz_)

            # The cross-updates should be done in parallel!
            a_y, b_x = extrapolate(a_y, b_x, ε, λ, C_zy, α_log, C_zy_), extrapolate(
                b_x, a_y, ε, λ, C_zx, β_log, C_zx_
            )

            # Update the measure weights and cost "matrices":
            k = k + 1
            α_log, β_log = α_logs[k], β_logs[k]
            if debias:
                C_xz, C_yz = C_xz_, C_yz_
            C_zx, C_zy = C_zx_, C_zy_

    torch.autograd.set_grad_enabled(True)

    #if last_extrapolation:
        ## Last extrapolation, to get the correct gradients:
        #if debias:
            #a_x = λ * softmin(ε, C_xz, (α_log + a_x / ε).detach())
            #b_y = λ * softmin(ε, C_yz, (β_log + b_y / ε).detach())

        # The cross-updates should be done in parallel!
        #a_y, b_x = λ * softmin(ε, C_zy, (α_log + b_x / ε).detach()), λ * softmin(
            #ε, C_zx, (β_log + a_y / ε).detach()
        #)
    return gamma_log.exp()/gamma_log.exp().sum()
    #if debias:
        #return a_x, b_y, a_y, b_x
    #else:
        #return None, None, a_y, b_x