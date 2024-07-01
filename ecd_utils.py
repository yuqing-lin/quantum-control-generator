# Adapted from the Echoed Conditional Displacement (ECD) Control project
# https://github.com/alec-eickbusch/ECD_control
# Based on the paper: Fast universal control of an oscillator with weak dispersive coupling to a qubit. (2022). Nature Physics, 18(12), 1464-1469. 

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Generate a Gaussian wave
def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)

# Rotate Gaussian wave
def rotate(theta, phi=0, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    ts = np.arange(len(wave)) * dt
    wave_interp = interp1d(x=ts, y=wave, kind="cubic")
    energy, _ = quad(wave_interp, a=0, b=(len(wave) - 1) * dt)
    amp = 1 / energy
    wave = (1 + 0j) * wave
    return (theta / 2.0) * amp * np.exp(1j * phi) * wave

# Displace cavity by alpha
def disp_gaussian(alpha, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    wave = (1 + 0j) * wave
    return (np.abs(alpha) / energy) * np.exp(1j * (np.pi / 2.0 + np.angle(alpha))) * wave

class FakePulse:
    def __init__(self, unit_amp, sigma, chop, detune=0):
        self.unit_amp = unit_amp
        self.sigma = sigma
        self.chop = chop
        self.detune = detune

    def make_wave(self, pad=False):
        wave = gaussian_wave(sigma=self.sigma, chop=self.chop)
        return np.real(wave), np.imag(wave)

class FakeStorage:
    def __init__(
        self,
        chi_kHz,
        # chi_prime_Hz=1.0,
        Ks_Hz=2.0,  # The Kerr effect strength
        epsilon_m_MHz=400.0,  # The maximum drive amplitude
        unit_amp=0.05,
        sigma=15,
        chop=4
    ):
        self.chi_kHz = chi_kHz
        # self.chi_prime_Hz = chi_prime_Hz
        self.Ks_Hz = Ks_Hz
        self.epsilon_m_MHz = epsilon_m_MHz

        self.displace = FakePulse(unit_amp=unit_amp, sigma=sigma, chop=chop)

        # Conversion between DAC and Hamiltonian drive amplitude
        # Scales the maximum amplitude of displacement to the desired units (MHz) and normalizes by the `unit_amp` and a factor of 2π
        disp = disp_gaussian(alpha=1.0, sigma=sigma, chop=chop, dt=1)
        self.epsilon_m_MHz = 1e3 * np.real(np.max(np.abs(disp))) / (2 * np.pi * unit_amp)

class FakeQubit:
    def __init__(self, unit_amp=0.5, sigma=6, chop=4, detune=0):
        self.pulse = FakePulse(unit_amp=unit_amp, sigma=sigma, chop=chop, detune=detune)
        # Conversion between DAC and Hamiltonian drive amplitude
        # The conversion factor scales the maximum amplitude of `rotate` to the desired units (MHz) and normalizes by the `unit_amp` and a factor of 2π.
        pi = rotate(np.pi, phi=0, sigma=sigma, chop=chop, dt=1)  # π rotation for the qubit
        self.Omega_m_MHz = 1e3 * np.real(np.max(np.abs(pi))) / (2 * np.pi * unit_amp)

# Interpolate data array
def interp(data_array, dt=1):
    ts = np.arange(0, len(data_array)) * dt
    return interp1d(ts, data_array, kind="cubic", bounds_error=False)

# Get flip indexes in qubit DAC pulse
def get_flip_idxs(qubit_dac_pulse):
    return find_peaks(qubit_dac_pulse, height=np.max(qubit_dac_pulse) * 0.975)[0]

# Solve nonlinear differential equation using finite difference method
def alpha_from_epsilon_nonlinear_finite_difference(epsilon_array, delta=0, Ks=0, kappa=0, alpha_init=0 + 0j):
    dt = 1
    alpha = np.zeros_like(epsilon_array)
    alpha[0] = alpha_init
    alpha[1] = alpha_init
    for j in range(1, len(epsilon_array) - 1):
        alpha[j + 1] = (
            2 * dt * (-1j * delta * alpha[j] + 2j * Ks * np.abs(alpha[j]) ** 2 * alpha[j] - (kappa / 2.0) * alpha[j] - 1j * epsilon_array[j])
            + alpha[j - 1]
        )
    return alpha

# Solve differential equations for qubit states g and e using scipy's solve_ivp
def alpha_from_epsilon_ge(epsilon_array, delta=0, chi=0, chi_prime=0, Ks=0, kappa=0, alpha_g_init=0 + 0j, alpha_e_init=0 + 0j):
    dt = 1
    t_eval = np.linspace(0, len(epsilon_array) * dt - dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    rtol = 1e-15
    atol = rtol

    dalpha_dt_g = lambda t, alpha: (-1j * delta * alpha + 2j * Ks * np.abs(alpha) ** 2 * alpha - (kappa / 2.0) * alpha - 1j * epsilon(t))
    alpha_g = solve_ivp(dalpha_dt_g, (0, len(epsilon_array) * dt - dt), y0=[alpha_g_init], method="RK45", t_eval=t_eval, rtol=rtol, atol=atol).y[0]

    if chi == 0 and chi_prime == 0 and alpha_g_init == alpha_e_init:
        alpha_e = alpha_g
    else:
        dalpha_dt_e = lambda t, alpha: (-1j * delta * alpha + 2j * Ks * np.abs(alpha) ** 2 * alpha - (kappa / 2.0) * alpha - 1j * epsilon(t) + 1j * (chi + 2 * chi_prime * np.abs(alpha) ** 2) * alpha)
        alpha_e = solve_ivp(dalpha_dt_e, (0, len(epsilon_array) * dt - dt), y0=[alpha_e_init], method="RK45", t_eval=t_eval, rtol=rtol, atol=atol).y[0]
    return alpha_g, alpha_e

# Solve differential equations for qubit states g and e using finite difference method
def alpha_from_epsilon_ge_finite_difference(epsilon_array, delta=0, chi=0, chi_prime=0, Ks=0, kappa=0, alpha_g_init=0 + 0j, alpha_e_init=0 + 0j):
    dt = 1
    alpha_g = np.zeros_like(epsilon_array)
    alpha_e = np.zeros_like(epsilon_array)
    alpha_g[0], alpha_g[1] = alpha_g_init, alpha_g_init
    alpha_e[0], alpha_e[1] = alpha_e_init, alpha_e_init
    for j in range(1, len(epsilon_array) - 1):
        alpha_g[j + 1] = (
            2 * dt * (-1j * delta * alpha_g[j] + 2j * Ks * np.abs(alpha_g[j]) ** 2 * alpha_g[j] - (kappa / 2.0) * alpha_g[j] - 1j * epsilon_array[j])
            + alpha_g[j - 1]
        )
        alpha_e[j + 1] = (
            2 * dt * (-1j * delta * alpha_e[j] + 2j * Ks * np.abs(alpha_e[j]) ** 2 * alpha_e[j] - (kappa / 2.0) * alpha_e[j] - 1j * epsilon_array[j] + 1j * (chi + 2 * chi_prime * np.abs(alpha_e[j]) ** 2) * alpha_e[j])
            + alpha_e[j - 1]
        )
    return alpha_g, alpha_e

# Get alpha trajectories for g and e states
def get_ge_trajectories(epsilon, delta=0, chi=0, chi_prime=0, Ks=0, kappa=0, flip_idxs=[], finite_difference=False):
    func = alpha_from_epsilon_ge_finite_difference if finite_difference else alpha_from_epsilon_ge
    f = lambda epsilon, alpha_g_init, alpha_e_init: func(epsilon, delta=delta, chi=chi, chi_prime=chi_prime, Ks=Ks, kappa=kappa, alpha_g_init=alpha_g_init, alpha_e_init=alpha_e_init)
    epsilons = np.split(epsilon, flip_idxs)
    alpha_g, alpha_e = [], []
    g_state = 0
    alpha_g_init, alpha_e_init = 0 + 0j, 0 + 0j

    for epsilon in epsilons:
        alpha_g_current, alpha_e_current = f(epsilon, alpha_g_init, alpha_e_init)
        if g_state == 0:
            alpha_g.append(alpha_g_current)
            alpha_e.append(alpha_e_current)
        else:
            alpha_g.append(alpha_e_current)
            alpha_e.append(alpha_g_current)
        alpha_g_init = alpha_e_current[-1]
        alpha_e_init = alpha_g_current[-1]
        g_state = 1 - g_state

    return np.concatenate(alpha_g), np.concatenate(alpha_e)

# This function uses pre-calibrated pulses to generate DAC pulses for a conditional displacement circuit.
# Note: It returns the DAC pulses directly, not the values of epsilon and Omega.
# Buffer time can be negative to perform the pi pulse while the cavity is being displaced.
# Conditional displacement is defined as:
# D(beta/2)|eXg| + D(-beta/2)|gXe|
def conditional_displacement(
    beta,
    alpha,
    storage,
    qubit,
    buffer_time=4,
    curvature_correction=True,
    pad=False,
    finite_difference=True,
    qubit_pulse_detuning=0
):
    beta = float(beta) if isinstance(beta, int) else beta
    alpha = float(alpha) if isinstance(alpha, int) else alpha
    chi = 2 * np.pi * 1e-6 * storage.chi_kHz
    chi_prime = 0
    Ks = 2 * np.pi * 1e-9 * storage.Ks_Hz
    delta = chi / 2.0
    epsilon_m = 2 * np.pi * 1e-3 * storage.epsilon_m_MHz
    alpha = np.abs(alpha)
    beta_abs = np.abs(beta)
    beta_phase = np.angle(beta)

    # Generate displacement and qubit pulse waves
    dr, di = storage.displace.make_wave(pad=False)
    d = storage.displace.unit_amp * (dr + 1j * di)
    pr, pi = qubit.pulse.make_wave(pad=False)
    detune = 2 * np.pi * qubit.pulse.detune + qubit_pulse_detuning
    
    if np.abs(detune) > 0:
        ts = np.arange(len(pr)) * 1e-9
        c_wave = (pr + 1j * pi) * np.exp(-2j * np.pi * ts * detune)
        pr, pi = np.real(c_wave), np.imag(c_wave)

    p = qubit.pulse.unit_amp * (pr + 1j * pi)

    # Only add buffer time at the final setp
    def construct_CD(alpha, tw, r, r0, r1, r2, buf=0):
        cavity_dac_pulse = r * np.concatenate([
            alpha * d * np.exp(1j * phase),
            np.zeros(tw),
            r0 * alpha * d * np.exp(1j * (phase + np.pi)),
            np.zeros(len(p) + 2 * buf),
            r1 * alpha * d * np.exp(1j * (phase + np.pi)),
            np.zeros(tw),
            r2 * alpha * d * np.exp(1j * phase),
        ])
        qubit_dac_pulse = np.concatenate([
            np.zeros(tw + 2 * len(d) + buf),
            p,
            np.zeros(tw + 2 * len(d) + buf),
        ])
        
        return cavity_dac_pulse, qubit_dac_pulse

    def integrated_beta_and_displacement(epsilon):
        flip_idx = int(len(epsilon) / 2)
        alpha_g, alpha_e = get_ge_trajectories(
            epsilon,
            delta=delta,
            chi=chi,
            chi_prime=chi_prime,
            Ks=Ks,
            flip_idxs=[flip_idx],
            finite_difference=finite_difference,
        )
        return np.abs(alpha_g[-1] - alpha_e[-1]), np.abs(alpha_g[-1] + alpha_e[-1])

    def ratios(alpha, tw):
        """
        Perform a minimization: given alpha and tw, find the ratio of middle and final pulses which:
        1. Returns the state to the middle after the first half;
        2. Returns the state to the middle after the entire sequence;
        3. (Bonus) Uses the same radius in the second half as in the first half.
        """
        n = np.abs(alpha) ** 2
        chi_effective = chi + 2 * chi_prime * n
        r = 1.0
        r0 = np.cos((chi_effective / 2.0) * tw)
        r1 = r0
        r2 = np.cos(chi_effective * tw)

        def cost(x):
            r = x[0]
            r0 = x[1]
            r1 = x[2]
            r2 = x[3]
            cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r0, r1, r2)
            epsilon = cavity_dac_pulse * epsilon_m
            flip_idx = int(len(epsilon) / 2)
            alpha_g, alpha_e = get_ge_trajectories(
                epsilon,
                delta=delta,
                chi=chi,
                chi_prime=chi_prime,
                Ks=Ks,
                flip_idxs=[flip_idx],
                finite_difference=finite_difference,
            )
            mid_disp = np.abs(alpha_g[flip_idx] + alpha_e[flip_idx])
            final_disp = np.abs(alpha_g[-1] + alpha_e[-1])
            first_radius = np.abs((alpha_g[int(flip_idx / 2)] + alpha_e[int(flip_idx / 2)]) / 2.0)
            second_radius = np.abs((alpha_g[int(3 * flip_idx / 2)] + alpha_e[int(3 * flip_idx / 2)]) / 2.0)
            return (np.abs(mid_disp) + np.abs(final_disp) + np.abs(first_radius - np.abs(alpha)) + np.abs(second_radius - np.abs(alpha)))

        result = fmin(cost, x0=[r, r0, r1, r2], ftol=1e-3, xtol=1e-3, disp=False)
        r = result[0]
        r0 = result[1]
        r1 = result[2]
        r2 = result[3]
        return r, r0, r1, r2

    phase = beta_phase + np.pi / 2.0
    n = np.abs(alpha) ** 2
    chi_effective = chi + 2 * chi_prime * n
    tw = int(np.abs(np.arcsin(beta_abs / (2 * alpha)) / chi_effective))
    # print("Waiting time between disp and qubit pulse: " + str(tw) + "ns")

    # Calculate ratios
    r, r0, r1, r2 = ratios(alpha, tw)
    cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r0, r1, r2)

    if curvature_correction:
        epsilon = cavity_dac_pulse * epsilon_m
        current_beta, current_disp = integrated_beta_and_displacement(epsilon)
        diff = np.abs(current_beta) - np.abs(beta)
        ratio = np.abs(current_beta) / np.abs(beta)
        if diff < 0:
            tw = int(tw * 1.5)
            ratio = 1.01
        tw_flag = True
        while np.abs(diff) / np.abs(beta) > 1e-3:
            if ratio > 1.0 and tw > 0 and tw_flag:
                tw = int(tw / ratio)
            else:
                tw_flag = False
                alpha = alpha / ratio
            r, r0, r1, r2 = ratios(alpha, tw)
            cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r0, r1, r2, buf=buffer_time)
            epsilon = cavity_dac_pulse * epsilon_m
            current_beta, current_disp = integrated_beta_and_displacement(epsilon)
            diff = np.abs(current_beta) - np.abs(beta)
            ratio = np.abs(current_beta) / np.abs(beta)
            
    # Add back in the buffer time to the pulse
    cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r0, r1, r2, buf=buffer_time)
    epsilon = cavity_dac_pulse * epsilon_m
    current_beta, current_disp = integrated_beta_and_displacement(epsilon)

    if pad:
        while len(cavity_dac_pulse) % 4 != 0:
            cavity_dac_pulse = np.pad(cavity_dac_pulse, (0, 1), mode="constant")
            qubit_dac_pulse = np.pad(qubit_dac_pulse, (0, 1), mode="constant")

    t_points_cavity = np.arange(len(cavity_dac_pulse))
    t_points_qubit = np.arange(len(qubit_dac_pulse))

    I_cavity_func = interp1d(t_points_cavity, np.real(storage.epsilon_m_MHz * cavity_dac_pulse), kind='cubic', fill_value="extrapolate")
    Q_cavity_func = interp1d(t_points_cavity, np.imag(storage.epsilon_m_MHz * cavity_dac_pulse), kind='cubic', fill_value="extrapolate")
    I_qubit_func = interp1d(t_points_qubit, np.real(qubit.Omega_m_MHz * qubit_dac_pulse), kind='cubic', fill_value="extrapolate")
    Q_qubit_func = interp1d(t_points_qubit, np.imag(qubit.Omega_m_MHz * qubit_dac_pulse), kind='cubic', fill_value="extrapolate")

    return I_cavity_func, Q_cavity_func, I_qubit_func, Q_qubit_func