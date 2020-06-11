from ode_composer.statespace_model import StateSpaceModel
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import DictionaryBuilder
from ode_composer.sbl import SBL
from ode_composer.measurements_generator import MeasurementsGenerator
import matplotlib.pyplot as plt
import numpy as np
from ode_composer.signal_preprocessor import (
    GPSignalPreprocessor,
    RHSEvalSignalPreprocessor,
    SplineSignalPreprocessor,
)


# define the model
states = {"x": "-y-z", "y": "x+a*y", "z": "b+x*z-z*c"}
parameters = {"a": 0.2, "b": 0.2, "c": 5.7}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print("Original Model:")
print(ss)

t_span = [0, 50]
y0 = {"x": 1.0, "y": 10.0, "z": 10.0}
data_points = 500
gp_interpolation_factor = 1
# simulate model
gm = MeasurementsGenerator(
    ss=ss, time_span=t_span, initial_values=y0, data_points=data_points
)
t, y = gm.get_measurements(SNR_db=20)
y_normfactor = np.mean(np.linalg.norm(y, axis=1))
# Fit a GP, find the derivative and std of each
gproc = GPSignalPreprocessor(
    t=t,
    y=y[0, :],
    selected_kernel="RBF",
    interpolation_factor=gp_interpolation_factor
)
x_samples, t_gp = gproc.interpolate(return_extended_time=True, noisy_obs=False)
gproc.calculate_time_derivative()
dxdt = gproc.dydt
dxdt_std = gproc.A_std

gproc_2 = GPSignalPreprocessor(
    t=t,
    y=y[1, :],
    selected_kernel="RatQuad",
    interpolation_factor=gp_interpolation_factor,
)
y_samples, _ = gproc_2.interpolate(noisy_obs=False)
gproc_2.calculate_time_derivative()
dydt = gproc_2.dydt
dydt_std = gproc_2.A_std

gproc_3 = GPSignalPreprocessor(
    t=t,
    y=y[2, :],
    selected_kernel="RatQuad",
    interpolation_factor=gp_interpolation_factor,
)
z_samples, _ = gproc_3.interpolate(noisy_obs=False)
gproc_3.calculate_time_derivative()
dzdt = gproc_3.dydt
dzdt_std = gproc_3.A_std

spline_1 = SplineSignalPreprocessor(t, y[0,:])
x_spline = spline_1.interpolate(t)
dx_spline = spline_1.calculate_time_derivative(t)

spline_2 = SplineSignalPreprocessor(t, y[1,:])
y_spline = spline_2.interpolate(t)
dy_spline = spline_2.calculate_time_derivative(t)

spline_3 = SplineSignalPreprocessor(t, y[2,:])
z_spline = spline_3.interpolate(t)
dz_spline = spline_3.calculate_time_derivative(t)

rhs_preprop = RHSEvalSignalPreprocessor(
    t=t, y=y, rhs_function=ss.get_rhs, states=states
)

plt.figure(0)
plt.plot(y[0,:])
plt.plot(y[1,:])
plt.plot(y[2,:])
plt.title("Observable states")
plt.legend(['x', 'y', 'z'])
plt.show()
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]
dx3 = dydt_rhs[2, :]

plt.figure(1)
plt.subplot(311)
plt.plot(t, y[0,:], label="Original")
plt.plot(t, x_spline, c='r', label="Spline")
plt.plot(t_gp, x_samples, c='orange', label="GP")
plt.fill_between(t_gp, x_samples - 2*dxdt_std, x_samples + 2*dxdt_std, alpha=0.2, color='k')
plt.ylabel("x")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(312)
plt.plot(t, y[1,:], label="RHS")
plt.plot(t, y_spline, c='r', label="Spline")
plt.plot(t_gp, y_samples, c='orange', label="GP")
plt.fill_between(t_gp, y_samples - 2*dydt_std, y_samples + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("y")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(313)
plt.plot(t, y[2,:], label="RHS diff")
plt.plot(t, z_spline, c='r', label="spline diff")
plt.plot(t_gp, z_samples, c='orange', label="GP diff")
plt.fill_between(t_gp, z_samples - 2*dzdt_std, z_samples + 2*dzdt_std, alpha=0.2, color='k')
plt.ylabel("derivative of z")
plt.xlabel("time")
plt.legend(loc="best")
plt.show()

plt.figure(1)
plt.subplot(311)
plt.plot(t, dx1, label="RHS diff")
plt.plot(t, dx_spline, c='r', label="Spline diff")
plt.plot(t_gp, dxdt, c='orange', label="GP diff")
plt.fill_between(t_gp, dxdt - 2*dxdt_std, dxdt + 2*dxdt_std, alpha=0.2, color='k')
plt.ylabel("derivative of x")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(312)
plt.plot(t, dx2, label="RHS diff")
plt.plot(t, dy_spline, c='r', label="Spline diff")
plt.plot(t_gp, dydt, c='orange', label="GP diff")
plt.fill_between(t_gp, dydt - 2*dydt_std, dydt + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("derivative of the y")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(313)
plt.plot(t, dx3, label="RHS diff")
plt.plot(t, dz_spline, c='r', label="spline diff")
plt.plot(t_gp, dzdt, c='orange', label="GP diff")
plt.fill_between(t_gp, dzdt - 2*dzdt_std, dzdt + 2*dzdt_std, alpha=0.2, color='k')
plt.ylabel("derivative of z")
plt.xlabel("time")
plt.legend(loc="best")
plt.show()

# step 1 define a dictionary
d_f = ["1", "x", "y", "z", "x*z", "y*x", "z*y"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
# associate variables with data
data = {"x": x_samples.T, "y": y_samples.T, "z": z_samples.T}
A = dict_builder.evaluate_dict(input_data=data)

# step 2 define an SBL problem
# with the Lin reg model and solve it
lambda_param = 10.0
lambda_param_1 = np.sqrt(np.linalg.norm(dxdt_std)) + 0.1
print(lambda_param_1)
sbl_x1 = SBL(
    dict_mtx=A, data_vec=dxdt, lambda_param=lambda_param_1, dict_fcns=d_f
)
sbl_x1.compute_model_structure()

lambda_param_2 = np.sqrt(np.linalg.norm(dydt_std)) + 0.1
print(lambda_param_2)
sbl_x2 = SBL(
    dict_mtx=A, data_vec=dydt, lambda_param=lambda_param_2, dict_fcns=d_f
)
sbl_x2.compute_model_structure()

lambda_param_3 =    (np.linalg.norm(dzdt_std)) + 0.1
print(lambda_param_3)
sbl_x3 = SBL(
    dict_mtx=A, data_vec=dzdt, lambda_param=lambda_param_3, dict_fcns=d_f
)
sbl_x3.compute_model_structure()


# step 4 reporting
# #build the ODE
zero_th = 1e-5

ode_model = StateSpaceModel(
    {
        "x": sbl_x1.get_results(zero_th=zero_th),
        "y": sbl_x2.get_results(zero_th=zero_th),
        "z": sbl_x3.get_results(zero_th=zero_th),
    },
    parameters=None,
)
print("Estimated ODE model:")
print(ode_model)
states = ["x", "y", "z"]
y0 = [1.0, 10.0, 10.0]
sol_ode = solve_ivp(fun=ode_model.get_rhs, t_span=t_span, t_eval=t, y0=y0, args=(states,))

plt.figure(2)
plt.subplot(221)
t_orig, y_orig = gm.get_measurements()
plt.plot(t_orig, y_orig[0, :], "b", label=r"$x(t)$")
plt.plot(t_orig, y_orig[1, :], "g", label=r"$y(t)$")
plt.plot(t_orig, y_orig[2, :], "r", label=r"$z(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("Observed state")
plt.title("Original")
plt.grid()
plt.subplot(222)
plt.plot(sol_ode.t, sol_ode.y[0, :], "b", label=r"$x(t)$")
plt.plot(sol_ode.t, sol_ode.y[1, :], "g", label=r"$(t)$")
plt.plot(sol_ode.t, sol_ode.y[2, :], "r", label=r"$z(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("Observed state")
plt.title("Estimation")
plt.grid()
plt.subplot(223)
plt.plot(y_orig[0, :], y_orig[1, :])
plt.xlabel(r"$x_1(t)$")
plt.ylabel(r"$x_2(t)$")
plt.subplot(224)
plt.plot(sol_ode.y[0, :], sol_ode.y[1, :])
plt.xlabel(r"$\hat{x}_1(t)$")
plt.ylabel(r"$\hat{x}_2(t)$")
plt.show()
