from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from ecd_utils import (
    conditional_displacement, FakeStorage, FakeQubit
)

# Pulse functions
def unitstep_t(start, length, t):
    return 0 + (start <= t <= start + length)

def gaussian_waveform(sigma, middle, t):
    return np.exp(-0.5 * ((t - middle) / sigma) ** 2)

def square_pulse(detune, amp, length, start, phase):
    return (
        lambda t: amp / 2 * unitstep_t(start, length, t) * np.cos(-detune * t + phase),
        lambda t: amp / 2 * unitstep_t(start, length, t) * np.sin(-detune * t + phase)
    )

def gaussian_pulse(detune, amp, length, start, chop, phase, subtract_offset=False):
    sigma = length / chop
    mid = start + length * 0.5
    ts = np.linspace(start, start + length, 100000)
    waveform = gaussian_waveform(sigma, mid, ts)
    offset = waveform[0] if subtract_offset else 0
    return (
        lambda t: amp / 2 * unitstep_t(start, length, t) * (gaussian_waveform(sigma, mid, t) - offset) / (1 - offset) * np.cos(-detune * t + phase),
        lambda t: amp / 2 * unitstep_t(start, length, t) * (gaussian_waveform(sigma, mid, t) - offset) / (1 - offset) * np.sin(-detune * t + phase)
    )

def rounded_square_pulse(detune, amp, length, start, rise_time, phase):
    def rounded_square(t):
        if t < start:
            return 0
        elif t < start + rise_time:
            return 0.5 * (1 - np.cos(np.pi * (t - start) / rise_time))
        elif t < start + length - rise_time:
            return 1
        elif t < start + length:
            return 0.5 * (1 + np.cos(np.pi * (t - start - length + rise_time) / rise_time))
        else:
            return 0
    return (
        lambda t: amp / 2 * rounded_square(t) * np.cos(-detune * t + phase),
        lambda t: amp / 2 * rounded_square(t) * np.sin(-detune * t + phase)
    )

# Calibration functions
def calibrate_square_unit_amp(length):
    return 2 / length

def calibrate_gaussian_unit_amp(length, chop):
    return 2 * chop / np.sqrt(2 * np.pi) / length / erf(chop / (2 * np.sqrt(2)))

def calibrate_rounded_square_unit_amp(length, rise_time):
    total_area = length - rise_time
    return 2 / total_area

def calibrate_square_pi_amp(length):
    return np.pi / length

def calibrate_gaussian_pi_amp(length, chop):
    return np.pi * chop / np.sqrt(2 * np.pi) / length / erf(chop / (2 * np.sqrt(2)))

def calibrate_rounded_square_pi_amp(length, rise_time):
    total_area = length - rise_time
    return np.pi / total_area

# Pulse generators
def generate_displacement(waveform, alpha, length, start, phase, chop=12, rise_time=1):
    amp = np.abs(alpha)
    alpha_phase = np.angle(alpha) + np.pi * 0.5
    phase += alpha_phase
    if waveform == "square":
        unit_amp = calibrate_square_unit_amp(length)
        return square_pulse(0, amp * unit_amp, length, start, phase)
    elif waveform == "gaussian":
        unit_amp = calibrate_gaussian_unit_amp(length, chop)
        return gaussian_pulse(0, amp * unit_amp, length, start, chop, phase)
    elif waveform == "rounded_square":
        unit_amp = calibrate_rounded_square_unit_amp(length, rise_time)
        return rounded_square_pulse(0, amp * unit_amp, length, start, rise_time, phase)

def generate_snap(chi, waveform, parameters, length, start, phase, chop=12, rise_time=1):
    snap_I = lambda t: 0
    snap_Q = lambda t: 0
    half_length = length * 0.5

    if waveform == "square":
        pi_amp = calibrate_square_pi_amp(half_length)
    elif waveform == "gaussian":
        pi_amp = calibrate_gaussian_pi_amp(half_length, chop)
    elif waveform == "rounded_square":
        pi_amp = calibrate_rounded_square_pi_amp(half_length, rise_time)

    for idx, theta in enumerate(parameters):
        if theta == 0:
            continue
        detune = -chi * idx
        theta_phase = np.pi - theta if theta > np.pi else theta

        if waveform == "square":
            I1, Q1 = square_pulse(detune, pi_amp, half_length, start, phase)
            I2, Q2 = square_pulse(detune, pi_amp, half_length, start + half_length, phase + theta_phase)
        elif waveform == "gaussian":
            I1, Q1 = gaussian_pulse(detune, pi_amp, half_length, start, chop, phase)
            I2, Q2 = gaussian_pulse(detune, pi_amp, half_length, start + half_length, chop, phase + theta_phase)
        elif waveform == "rounded_square":
            I1, Q1 = rounded_square_pulse(detune, pi_amp, half_length, start, rise_time, phase)
            I2, Q2 = rounded_square_pulse(detune, pi_amp, half_length, start + half_length, rise_time, phase + theta_phase)

        snap_I = lambda t, snap_I=snap_I, I1=I1, I2=I2: snap_I(t) + I1(t) + I2(t)
        snap_Q = lambda t, snap_Q=snap_Q, Q1=Q1, Q2=Q2: snap_Q(t) + Q1(t) + Q2(t)

    return snap_I, snap_Q

def generate_ecd(beta, chi, length, start, phase, unit_amp, alpha_CD=30, buffer_time=1, curvature_correction=False, finite_difference=True):
    # chi_prime_Hz = 1.0
    # Ks_Hz = 2.0  # The Kerr effect strength
    # epsilon_m_MHz = 400.0  # The maximum drive amplitude
    # sigma = 15
    # chop = 4
    # max_dac = 0.6  # The maximum DAC amplitude
    storage = FakeStorage(chi_kHz=chi, unit_amp=unit_amp)
    qubit = FakeQubit()

    I_cavity_func, Q_cavity_func, I_qubit_func, Q_qubit_func = conditional_displacement(
        beta,
        alpha=alpha_CD,
        storage=storage,
        qubit=qubit,
        buffer_time=buffer_time,
        curvature_correction=curvature_correction,
        finite_difference=finite_difference,
        system=None
    )

    # Determine the original time range
    original_t_max = max(I_cavity_func.x[-1], Q_cavity_func.x[-1], I_qubit_func.x[-1], Q_qubit_func.x[-1])
    # Normalize the time to fit within the desired length
    scale_factor = length / original_t_max
    def normalize_time(func):
        return lambda t: func(t / scale_factor)
    
    I_cavity_norm = normalize_time(I_cavity_func)
    Q_cavity_norm = normalize_time(Q_cavity_func)
    I_qubit_norm = normalize_time(I_qubit_func)
    Q_qubit_norm = normalize_time(Q_qubit_func)

    phase_factor = np.exp(1j * phase)
    I_transmon = lambda t: np.real(phase_factor * I_qubit_norm(t - start)) if start <= t < start + length else 0
    Q_transmon = lambda t: np.imag(phase_factor * Q_qubit_norm(t - start)) if start <= t < start + length else 0
    I_cavity = lambda t: np.real(phase_factor * I_cavity_norm(t - start)) if start <= t < start + length else 0
    Q_cavity = lambda t: np.imag(phase_factor * Q_cavity_norm(t - start)) if start <= t < start + length else 0

    return I_transmon, Q_transmon, I_cavity, Q_cavity

# Control signal plotting function
def plot_control_signals(tlist, transmon_I, transmon_Q, cavity_I, cavity_Q):
    transmon_I_vals = [transmon_I(t) for t in tlist]
    transmon_Q_vals = [transmon_Q(t) for t in tlist]
    cavity_I_vals = [cavity_I(t) for t in tlist]
    cavity_Q_vals = [cavity_Q(t) for t in tlist]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(tlist, transmon_I_vals, label='Transmon I')
    plt.plot(tlist, transmon_Q_vals, label='Transmon Q')
    plt.title('Transmon Drive Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tlist, cavity_I_vals, label='Cavity I')
    plt.plot(tlist, cavity_Q_vals, label='Cavity Q')
    plt.title('Cavity Drive Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Circuit Builder
def build_circuit_and_generate_signal(Nc, chi, pulse_params, plot_signals=False):
    """
    Build the circuit and generate control signals for the given pulse parameters.
    
    Parameters:
    chi (float): The interaction strength.
    pulse_params (list): List of dictionaries containing pulse parameters. Each dictionary has the keys:
                         - type (str): 'snap' or 'displacement'.
                         - parameters (list or complex): Parameters for the pulse.
                         - t_i (float): Initial time for the pulse.
                         - t_f (float): Final time for the pulse.
                         - waveform (str): 'square', 'gaussian', or 'rounded_square'.
                         - chop (int, optional): Chop factor for Gaussian pulses.
                         - rise_time (float, optional): Rise time for rounded square pulses.
                         - cavity_index (int, optional): Index of the cavity for displacement.
    plot_signals (bool, optional): Whether to plot the control signals. Default is False.
    
    Returns:
    dict: Control signals for the system.
    """
    transmon_I = lambda t: 0
    transmon_Q = lambda t: 0
    cavity_I_list = [lambda t: 0 for _ in range(Nc)]
    cavity_Q_list = [lambda t: 0 for _ in range(Nc)]

    for gate in pulse_params:
        t_i = gate["t_i"]
        t_f = gate["t_f"]
        length = t_f - t_i

        phase = gate.get("phase", 0)

        if gate["type"] == "snap":
            I, Q = generate_snap(chi, gate["waveform"], gate["parameters"], length, t_i, gate.get("chop", 6), gate.get("rise_time", 1), phase)
            transmon_I = lambda t, transmon_I=transmon_I, I=I: transmon_I(t) + I(t)
            transmon_Q = lambda t, transmon_Q=transmon_Q, Q=Q: transmon_Q(t) + Q(t)
        elif gate["type"] == "displacement":
            I, Q = generate_displacement(gate["waveform"], gate["parameter"], length, t_i, gate.get("chop", 6), gate.get("rise_time", 1), phase)
            cavity_index = int(gate["cavity_index"]) - 1  # Adjust for 1-based indexing
            cavity_I_list[cavity_index] = lambda t, cavity_I=cavity_I_list[cavity_index], I=I: cavity_I(t) + I(t)
            cavity_Q_list[cavity_index] = lambda t, cavity_Q=cavity_Q_list[cavity_index], Q=Q: cavity_Q(t) + Q(t)
        elif gate["type"] == "ecd":
            I_transmon, Q_transmon, I_cavity, Q_cavity = generate_ecd(gate["parameter"], chi, length, t_i, phase, gate["unit_amp"])
            transmon_I = lambda t, transmon_I=transmon_I, I_trans=I_transmon: transmon_I(t) + I_trans(t)
            transmon_Q = lambda t, transmon_Q=transmon_Q, Q_trans=Q_transmon: transmon_Q(t) + Q_trans(t)
            cavity_index = int(gate["cavity_index"]) - 1  # Adjust for 1-based indexing
            cavity_I_list[cavity_index] = lambda t, cavity_I=cavity_I_list[cavity_index], I_cav=I_cavity: cavity_I(t) + I_cav(t)
            cavity_Q_list[cavity_index] = lambda t, cavity_Q=cavity_Q_list[cavity_index], Q_cav=Q_cavity: cavity_Q(t) + Q_cav(t)

    if plot_signals:
        sequence_length = pulse_params[-1]["t_f"]
        tlist = np.linspace(0, sequence_length, 100000)
        for i in range(Nc):
            plot_control_signals(tlist, transmon_I, transmon_Q, cavity_I_list[i], cavity_Q_list[i])

    return {
        "transmon_I": transmon_I,
        "transmon_Q": transmon_Q,
        **{f"cavity_{i+1}_I": cavity_I for i, cavity_I in enumerate(cavity_I_list)},
        **{f"cavity_{i+1}_Q": cavity_Q for i, cavity_Q in enumerate(cavity_Q_list)},
    }