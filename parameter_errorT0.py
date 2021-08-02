from adnls import fit
import autograd.numpy as np

# The model used for nonlinear regression
def model(t, p):
    y = p[0] * (1-p[1]*t**(-1/3))
    return y

# The residual function used by the `fit` class
def res(p, extra_pars):
    return model(extra_pars["t"], p) - extra_pars["data"]

extra_pars = {}
extra_pars["t"] = np.array([6, 12, 14, 17, 19, 22, 24, 27, 28, 32, 39, 45, 46, 48, 51, 54, 55, 59, 60, 63, 75, 84, 85, 89, 90, 93, 103, 107, 110, 112, 115, 116, 126, 127, 133, 138, 139, 140, 141, 159, 175, 186, 198, 208, 209])
extra_pars["data"] = np.array([2.4, 12.4 ,  13.2 ,  14.7 ,  17.8 ,  14.2 ,  17.9 ,  15.6 ,  16.5 ,  19.8 ,  20 ,  22.1 ,  19.7 ,  18.9 ,  19.7 ,  20.5 ,  19.4 ,  19.4 ,  19.7 ,  19.6 ,  20 ,  23.2 ,  22.5 ,  22.3 ,  23.6 ,  22.5 ,  22.8 ,  23.5 ,  24 ,  23.2 ,  22.9 ,  23.9 ,  23.2 ,  24.3 ,  25.2 ,  26.2 ,  25.9 ,  26.3 ,  26.5 ,  21.9 ,  24 ,  26.9 ,  29.3 ,  27.2 ,  26.7])
x_best = np.array((35.5808, 1.57907), dtype=np.float64)

this_fit = fit(
    res_func=res,
    bestfit_pars=x_best,
    extra_pars=extra_pars,
)  # create instance of fit class
print("Hessian")
print(this_fit.get_H(x_best))
print(this_fit.get_H())
print("Jacobian")
print(this_fit.get_J(x_best))
print("Residual")
print(this_fit.residual())
print(this_fit.residual(x_best))
print("Vcov matrix")
print(this_fit.get_vcov())
print("Sd of best-fit paramters")
print(this_fit.get_sd_bf())

