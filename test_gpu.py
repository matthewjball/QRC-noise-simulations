from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
import time

# Create a slightly larger circuit
from qiskit import QuantumCircuit
qc = QuantumCircuit(8)
for _ in range(100):
    qc.h(range(8))
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(4, 5)
    qc.cx(6, 7)

backend = AerSimulator(method='density_matrix', device='GPU')
backend_noiseless = AerSimulator(method='density_matrix', device='GPU')

# Warm up GPU
_ = backend.run(transpile(qc, backend)).result()

# Measure timing
times = []
for _ in range(5):
    start = time.time()
    result = backend.run(transpile(qc, backend)).result()
    times.append(time.time() - start)

print(f"Average GPU simulation time: {sum(times)/len(times):.3f}s ± {max(times)-min(times):.3f}s")
print(f"Devices used: {backend.available_devices()}")