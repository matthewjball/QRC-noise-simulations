import numpy as np


T1_T2_MAP = {
    0.0001: {'t1': 800e-6, 't2': 1120e-6},   # Low noise
    0.0005: {'t1': 160e-6, 't2': 224e-6},    # Below-average noise
    0.001:  {'t1': 80e-6, 't2': 112e-6},     # Realistic noise on IBM device
    0.003:  {'t1': 27e-6, 't2': 38e-6},      # High-noise 
}

# Gate times (seconds)
GATE_TIME_1Q = 57e-9   # Single-qubit gates
GATE_TIME_2Q = 533e-9  # Two-qubit CX gates

def calc_error_rate(t1, t2, gate_time):
    """Calculate actual thermal relaxation error probability"""
    gamma = 1 - np.exp(-gate_time / t1)      # Amplitude damping
    lambda_dephase = np.exp(-gate_time / t2) - np.exp(-gate_time / (2 * t1))  # Pure dephasing
    epsilon = 1 - np.exp(-gate_time / t1) * np.exp(-gate_time / t2)
    return epsilon

# Validation
for p_target, params in T1_T2_MAP.items():
    t1, t2 = params['t1'], params['t2']
    eps_1q = calc_error_rate(t1, t2, GATE_TIME_1Q)
    eps_2q = calc_error_rate(t1, t2, GATE_TIME_2Q)
    
    print(f"p={p_target:.4f}: T1={t1*1e6:.0f}µs, T2={t2*1e6:.0f}µs")
    print(f"  → 1Q error: {eps_1q:.6f} ({eps_1q*100:.4f}%), 2Q error: {eps_2q:.6f} ({eps_2q*100:.4f}%)")


# This python code validates relaxation and decoherence values from given error probabilities.
# Note that the error rate is scaled by the gate time 