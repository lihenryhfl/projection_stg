import numpy as np
import torch


def regress(X, y):
    # takes X and y as input, and returns theta
    # from X theta = y
    XTX_inv = torch.pinverse(torch.mm(X.T, X))
    theta = torch.mm(torch.mm(XTX_inv, X.T), y)
    return theta


def sample_z(mu, n_est):
    # get shape of theta
    d1, d2 = mu.shape

    # ones n zeros
    ones = torch.ones(size=(n_est, d1, d2))
    zeros = torch.zeros(size=(n_est, d1, d2))

    epsilon = 0.5 * torch.randn(size=(n_est, d1, d2))
    tmp = epsilon + 0.5 + torch.unsqueeze(mu, 0)

    return torch.max(zeros, torch.min(tmp, ones))


def regress_mu(X, y, mu, n_est=1000):
    # estimate E[Z Z^T]
    Zs = sample_z(mu, n_est=n_est)
    EZZT = torch.mean(torch.einsum("nij,nkj->nik", Zs, Zs), dim=0)
    EZ = torch.mean(Zs, dim=0)

    # perform regression
    XTX = torch.mm(X.T, X)
    XTX_EZZT_inv = torch.pinverse(XTX * EZZT)
    theta = torch.mm(XTX_EZZT_inv, (X.T @ y) * EZ)

    return theta


def det_z(mu):
    # get shape of theta
    d1, d2 = mu.shape

    # ones n zeros
    ones = torch.ones(size=(d1, d2))
    zeros = torch.zeros(size=(d1, d2))

    tmp = torch.unsqueeze(mu, 0)

    return torch.max(zeros, torch.min(tmp, ones))


# define risk function


def risk(X, y, theta_hat, mu, lam=1, n_est=1000):
    # obtain another estimate of Z
    Zs = sample_z(mu, n_est=n_est)

    theta_Zs = torch.unsqueeze(theta_hat, 0) * Zs
    assert theta_Zs.shape == Zs.shape

    X_theta_Zs = torch.einsum("ij,njk->nik", X, theta_Zs)

    # subtract from y
    errors = torch.sum((X_theta_Zs - torch.unsqueeze(y, 0)) ** 2, dim=-1)
    error = torch.mean(errors)

    # compute the regularization term
    phi = torch.distributions.normal.Normal(loc=0, scale=1).cdf((mu + 0.5) * 2)
    reg = torch.sum(phi)

    return error + lam * reg, reg


def stg_proj(
    X, y, lam, n_iter=3000, n_est=50, learning_rate=1e-3, print_step=1000, verbose=False
):

    mu = torch.zeros(size=[X.shape[1], 1], requires_grad=True)
    no_theta_grad = True
    optimizer = torch.optim.Adam([mu], lr=learning_rate)
    for i in range(n_iter):

        if no_theta_grad:
            with torch.no_grad():
                theta_hat_mu = regress_mu(X, y, mu, n_est=n_est)
        elif torch.isnan(sum(mu)) == False:
            theta_hat_mu = regress_mu(X, y, mu, n_est=n_est)

        # compute loss
        loss, reg = risk(X, y, theta_hat_mu, mu, n_est=n_est, lam=lam)

        if verbose and i % print_step == 0:
            print("iter %d loss %0.5f reg %0.5f" % (i, loss, reg))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    flag = True

    theta_hat_mu = regress_mu(X, y, mu, n_est=n_est)
    theta_hat_mu = theta_hat_mu * det_z(mu)
    return theta_hat_mu
