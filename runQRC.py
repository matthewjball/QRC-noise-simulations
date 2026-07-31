import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.stats import unitary_group

import itertools, random, sys, argparse
import qiskit_aer.noise as noise

from qiskit import *
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, UnitaryGate
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, Pauli, SparsePauliOp, Operator
from qiskit_aer import AerSimulator

from qiskit.providers.fake_provider import GenericBackendV2

# from qiskit.test.mock import *

class QuantumCircQiskit:
    def __init__(self, gates_name, num_gates=50,nqbits=8,observables_type = 'fidelity',
                 err_type='depolarizing', err_p1=0.001, err_p2=0.01, err_idle=0.00001):
        
        self.num_gates = num_gates
        self.gates_name = gates_name
        self.observables_type = observables_type
        self.gates_set = []
        self.qubits_set = []
        self.nqbits=nqbits
        self.err_type = err_type

        noise_model, basis_gates = self.get_noise_model(err_type=err_type, p1=err_p1, p2=err_p2, p_idle=err_idle)
            
        self.noise_model = noise_model
        self.basis_gates = basis_gates

        if self.gates_name=='G1':
            gates = ['CNOT', 'H', 'X']
        if self.gates_name=='G2':
            gates = ['CNOT', 'H', 'S']
        if self.gates_name=='G3':
            gates = ['CNOT', 'H', 'T']  
        if self.gates_name=='HT':
            gates = ['H', 'T']  
        if self.gates_name=='Toffoli':
            gates = ['CCX'] 


        qubit_idx = list(range(self.nqbits))
        # Store gates
        if self.gates_name in ['G1', 'G2', 'G3', 'Toffoli']:
            for i in range(self.num_gates):
                # Select random gate
                gate = random.sample(gates,1)[0] 
                self.gates_set.append(gate)
                if gate=='CNOT':
                    # Select qubit 1 and 2 (different qubits)
                    qbit1 = random.sample(qubit_idx,1)[0]
                    qubit_idx2 = qubit_idx.copy()
                    qubit_idx2.remove(qbit1)
                    qbit2 = random.sample(qubit_idx2,1)[0]
                    self.qubits_set.append([qbit1, qbit2])
                elif gate=='CCX':
                    # Select qubit 1, 2 and 3 (different qubits)
                    qbit1 = random.sample(qubit_idx,1)[0]
                    qubit_idx2 = qubit_idx.copy()
                    qubit_idx2.remove(qbit1)
                    qbit2 = random.sample(qubit_idx2,1)[0]
                    qubit_idx3 = qubit_idx2.copy()
                    qubit_idx3.remove(qbit2)
                    qbit3 = random.sample(qubit_idx3,1)[0]
                    self.qubits_set.append([qbit1, qbit2, qbit3])
                else:
                    # Select qubit
                    qbit = random.sample(qubit_idx,1)[0]
                    self.qubits_set.append([qbit])
        elif self.gates_name=='D2':
            qubit_idx = list(range(self.nqbits))
            self.qubits_set = list(itertools.combinations(qubit_idx, 2))
            self.phis = np.random.uniform(0, 2*np.pi, size=(len(self.qubits_set), 2**2))
        elif self.gates_name=='D3':
            qubit_idx = list(range(self.nqbits))
            self.qubits_set = list(itertools.combinations(qubit_idx, 3))
            self.phis = np.random.uniform(0, 2*np.pi, size=(len(self.qubits_set), 2**3))
        elif self.gates_name=='Dn':
            self.phis = np.random.uniform(0, 2*np.pi, size=(2**self.nqbits))
        elif self.gates_name=='MG':
            for i in range(self.num_gates):
                G = self.matchgate()
                self.gates_set.append(G)
                qbit1 = random.sample(qubit_idx,1)[0]
                qubit_idx2 = qubit_idx.copy()
                qubit_idx2.remove(qbit1)
                qbit2 = random.sample(qubit_idx2,1)[0]
                self.qubits_set.append([qbit1, qbit2])

                
    def initialization(self, initial_state):
        # 1. INITIALIZATION
        # Define initial state
        initial_state = initial_state.round(6)
        initial_state/=np.sqrt(np.sum(initial_state**2))
    
        dim = len(initial_state)

        # Create matrix with initial_state as first column
        A = np.zeros((dim, dim), dtype=np.complex128)
        A[:, 0] = initial_state
    
        # Fill remaining columns with random vectors
        for i in range(1, dim):
            A[:, i] = np.random.randn(dim) + 1j * np.random.randn(dim)
    
        # QR decomposition guarantees unitary Q
        Q, R = np.linalg.qr(A)
        
        U = UnitaryGate(Q, label='unitary')
        
        return U

    def apply_G_gates(self, qc):
        # Apply random gates to random qubits
        for i in range(self.num_gates):
            # Select random gate
            # Select random gate
            gate = self.gates_set[i]
            if gate=='CNOT': # For 2-qubit gates
                # Select qubit 1 and 2 (different qubits)
                qbit1, qbit2 = self.qubits_set[i]
                # Apply gate to qubits
                qc.cx(qbit1, qbit2) 
                # Appply identity operator to all idle qubits if we use an ide noise model
                if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                    qubit_idx = list(range(self.nqbits))
                    qubit_idx.remove(qbit1)
                    qubit_idx.remove(qbit2)
                    # Apply identity gates to other gates
                    for qbit in qubit_idx:
                        qc.id(qbit)
            if gate=='CCX':
                # Select qubit 1, 2 and 3 (different qubits)
                qbit1, qbit2, qbit3 = self.qubits_set[i]
                # Apply gate to qubits
                qc.ccx(qbit1, qbit2, qbit3) 
                if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                    qubit_idx = list(range(self.nqbits))
                    qubit_idx.remove(qbit1)
                    qubit_idx.remove(qbit2)
                    qubit_idx.remove(qbit3)
                    # Apply identity gates to other gates
                    for qbit in qubit_idx:
                        qc.id(qbit)
            else: # For 1-qubit gates
                # Select qubit
                qbit = self.qubits_set[i][0]
                if gate=='X':# Apply gate
                    qc.x(qbit) 
                if gate=='S':
                    qc.s(qbit) 
                if gate=='H':
                    qc.h(qbit) 
                if gate=='T':
                    qc.t(qbit) 
                # Appply identity operator to all idle qubits if we use an ide noise model
                if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                    qubit_idx = list(range(self.nqbits))
                    qubit_idx.remove(qbit)
                    # Apply identity gates to other gates
                    for qbit in qubit_idx:
                        qc.id(qbit)
                
    
    def apply_matchgates(self, qc):
        for i in range(self.num_gates):
            gate = self.gates_set[i]
            qbit1, qbit2 = self.qubits_set[i]
            qc.unitary(gate, [qbit1, qbit2], label='MG')
            # Appply identity operator to all idle qubits if we use an ide noise model
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                qubit_idx = list(range(self.nqbits))
                qubit_idx.remove(qbit1)
                qubit_idx.remove(qbit2)
                # Apply identity gates to other gates
                for qbit in qubit_idx:
                    qc.id(qbit)
            
    def matchgate(self):
        A = unitary_group.rvs(2)
        B = unitary_group.rvs(2)
        detA = np.linalg.det(A)
        detB = np.linalg.det(B)
        B = B/np.sqrt(detB)*np.sqrt(detA)
        G = np.array([[A[0,0],0,0,A[0,1]],[0,B[0,0], B[0,1],0],
                      [0,B[1,0],B[1,1],0],[A[1,0],0,0,A[1,1]]])
        return G
    
    def apply_Dn(self, qc):
        # Apply Dn gate
        diagonals = np.exp(1j*self.phis)
        qc = qc.compose(Diagonal(diagonals))
        
    def apply_D2(self, qc):
        i=0
        for pair in self.qubits_set:
            # Apply D2 gate
            diagonals = np.diag(np.exp(1j*self.phis[i]))
            D2 = UnitaryGate(diagonals)
            qc.append(D2, [pair[0], pair[1]])
            i+=1
            # Appply identity operator to all idle qubits if we use an ide noise model
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                qubit_idx = list(range(self.nqbits))
                qubit_idx.remove(pair[0])
                qubit_idx.remove(pair[1])
                # Apply identity gates to other gates
                for qbit in qubit_idx:
                    qc.id(qbit)
            
    def apply_D3(self, qc):
        i=0
        for pair in self.qubits_set:
            # Apply D3 gate
            diagonals = np.diag(np.exp(1j*self.phis[i]))
            D3 = UnitaryGate(diagonals)
            qc.append(D3, [pair[0], pair[1], pair[2]])
            i+=1
            # Appply identity operator to all idle qubits if we use an ide noise model
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                qubit_idx = list(range(self.nqbits))
                qubit_idx.remove(pair[0])
                qubit_idx.remove(pair[1])
                qubit_idx.remove(pair[2])
                # Apply identity gates to other gates
                for qbit in qubit_idx:
                    qc.id(qbit)
                    
    def get_noise_model(self, err_type='depolarizing', p1=0.001, p2=0.01, p_idle=0.0001):
        # Error probabilities: p1=1-qubit gate, p2=2-qubit gate
        if err_type=='depolarizing' or err_type=='depolarizing_idle':
            # Depolarizing quantum errors
            error_1 = noise.depolarizing_error(p1, 1)
            error_2 = noise.depolarizing_error(p2, 2)
            error_idle = noise.depolarizing_error(p_idle, 1)
        elif err_type=='phase_damping' or err_type=='phase_damping_idle':
            # Depolarizing quantum errors
            error_1 = noise.phase_damping_error(p1, 1)
            error_2 = noise.phase_damping_error(p2, 2)
            error_2 = error_1.tensor(error_2)
            error_idle = noise.phase_damping_error(p_idle, 1)
        elif err_type=='amplitude_damping' or err_type=='amplitude_damping_idle':
            # Construct the error
            error_1 = noise.amplitude_damping_error(p1)
            error_2 = noise.amplitude_damping_error(p2)
            error_2 = error_1.tensor(error_2)
            error_idle = noise.amplitude_damping_error(p_idle, 1)

        elif err_type=='thermal_relaxation' or err_type=='thermal_relaxation_idle':
            # do something
            print('thermal')

        elif err_type=='none':
            return None, None
        else:
            raise ValueError('Error type not supported', err_type)
        # Add errors to noise model
        noise_model = noise.NoiseModel()
        
        noise_model.add_all_qubit_quantum_error(error_1, ['x','h','ry','rz','u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        noise_model.add_basis_gates('unitary')
        if err_type=='depolarizing_idle' or err_type=='amplitude_damping_idle' or err_type=='phase_damping_idle':
            noise_model.add_all_qubit_quantum_error(error_idle, ['id'])

        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        return noise_model, basis_gates
        
    def get_observables(self):
        observables = []
        name_gate=''
        for i in range(self.nqbits):
            name_gate+= 'I' 
        for i in range(self.nqbits):
            # X
            op_nameX = name_gate[:i] + 'X' + name_gate[(i+1):]
            obs = SparsePauliOp(Pauli(op_nameX))
            observables.append(obs)
            # Y
            op_nameY = name_gate[:i] + 'Y' + name_gate[(i+1):]
            obs = SparsePauliOp(Pauli(op_nameY))
            observables.append(obs)
            # Z
            op_nameZ = name_gate[:i] + 'Z' + name_gate[(i+1):]
            obs = SparsePauliOp(Pauli(op_nameZ))
            observables.append(obs)
        return observables

    def run_circuit(self, initial_state):

        # 1. INITIALIZATION
        U = self.initialization(initial_state)
        
        qc =  QuantumCircuit(self.nqbits)
        qc.append(U, list(range(self.nqbits)))
        # 2. DEFINE RANDOM CIRCUIT
        if self.gates_name in ['G1', 'G2', 'G3', 'Toffoli']:
            self.apply_G_gates(qc)
        elif self.gates_name=='D2':
            self.apply_D2(qc)
        elif self.gates_name=='D3':
            self.apply_D3(qc)
        elif self.gates_name=='Dn':
            self.apply_Dn(qc)
        elif self.gates_name=='MG':
            self.apply_matchgates(qc)
        else:
            print('Unknown gate')

        # 3. DEFINE OBSERVABLES
        # Define observables to measure
        if self.observables_type=='single' or self.observables_type=='all':
            observables = self.get_observables()



        # 4. RUN CIRCUIT
        results = []
        results_noiseless = []
        
        qc_noiseless = qc.copy()

        backend = AerSimulator(method='density_matrix', noise_model=self.noise_model)
        qc.save_density_matrix()

        optimization_level = 0 if self.err_type in ['depolarizing_idle', 'amplitude_damping_idle', 'phase_damping_idle'] else 1
        transpiled_qc = transpile(qc, backend, optimization_level=optimization_level)
        job = backend.run(transpiled_qc)

        result = job.result()

        # 4.1 Debug info
        print(f'\nDEBUG INFO')
        print(result)
        print(f'\n')

        dm_data = result.data(0)["density_matrix"]
        dm = DensityMatrix(dm_data)
        qc_state = dm.data
        
        if self.observables_type=='fidelity':
            backend_noiseless = AerSimulator(method='density_matrix')
            qc_noiseless.save_density_matrix()

            job_noiseless = backend.run(transpile(qc_noiseless, backend_noiseless))
            result_noiseless = job_noiseless.result()
            
            dm_data_noiseless = result_noiseless.data(0)["density_matrix"]
            dm_noiseless = DensityMatrix(dm_data_noiseless)
            qc_state_noiseless = dm_noiseless.data

            fidelity = state_fidelity(qc_state,qc_state_noiseless)
            return np.array(qc_state), np.array(qc_state_noiseless), fidelity
        
        if self.observables_type=='all':
            backend_noiseless = AerSimulator(method='density_matrix')
            qc_noiseless.save_density_matrix()

            job_noiseless = backend.run(transpile(qc_noiseless, backend_noiseless))
            result_noiseless = job_noiseless.result()

            dm_data_noiseless = result_noiseless.data(0)["density_matrix"]
            dm_noiseless = DensityMatrix(dm_data_noiseless)
            qc_state_noiseless = dm_noiseless.data

            fidelity = state_fidelity(qc_state,qc_state_noiseless)

            for obs in observables:
                obs_mat = obs.to_matrix()
                expect = np.inner(np.conjugate(qc_state), obs_mat.dot(qc_state)).real
                results.append(expect)

                expect_noiseless = np.inner(np.conjugate(qc_state_noiseless), obs_mat.dot(qc_state_noiseless)).real
                results_noiseless.append(expect_noiseless)

            return np.array(qc_state), np.array(qc_state_noiseless), fidelity, np.array(results), np.array(results_noiseless)
        
        if self.observables_type=='single':
            for obs in observables:
                obs_mat = obs.to_matrix()
                expect = np.inner(np.conjugate(qc_state), obs_mat.dot(qc_state)).real
                results.append(expect)

            return np.array(results)
        else:
            return qc_state
        
        
# Read user argument (number of gates and gates set)
parser = argparse.ArgumentParser()

parser.add_argument("--num_gates", default=10, help="Number of gate operations for simulations - default 10")
parser.add_argument("--gate_set", default='G3', help="Set of gates for simulations - see README for options")
parser.add_argument("--observables_type", default='all', help="Type of observable data reported - see README for options")
parser.add_argument("--err_type", required=True,help="Error type simulated - see README for options")
parser.add_argument("--err_p1", required=True, help="Probability of error on first run through")
parser.add_argument("--err_p2", required=True, help="Probability of error on second run through")

args = parser.parse_args()

num_gates = int(args.num_gates)
gate_set = str(args.gate_set)
observables_type = str(args.observables_type)
err_type = str(args.err_type)
err_p1 = float(args.err_p1)
err_p2 = float(args.err_p2)

print('Num gates: ', num_gates, ' gate_set: ', gate_set, ' observables_type:', observables_type,
      ' err_type: ', err_type, ' err_p1: ', err_p1, ' err_p2: ', err_p2  )

# Read data
with open('training_data/ground_states_LiH.npy', 'rb') as f:
        ground_states = np.load(f)

for j in range(5):
    # Run circuit for all values of ground states:
    
    num_states =ground_states.shape[0]
    qc = QuantumCircQiskit(gate_set, num_gates=num_gates,nqbits=8,observables_type = observables_type,
                      err_type=err_type, err_p1=err_p1, err_p2=err_p2)
    if observables_type=='single':
        obs_res = []
        for i in range(num_states):
            res = qc.run_circuit(ground_states[i])
            obs_res.append(res)
        obs_res = np.array(obs_res)    
    elif observables_type=='fidelity':
        state_noise_list, state_noiseless_list, fidelity_list = [],[],[]
        for i in range(num_states):
            state_noise, state_noiseless, fidelity = qc.run_circuit(ground_states[i])
            state_noise_list.append(state_noise)
            state_noiseless_list.append(state_noiseless)
            fidelity_list.append(fidelity)
    elif observables_type=='all':
        fidelity_list, obs_res, obs_noiseless = [],[],[]
        for i in range(num_states):
            _, _, fidelity, res, res_noiseless = qc.run_circuit(ground_states[i])
            fidelity_list.append(fidelity)
            obs_res.append(res)
            obs_noiseless.append(res_noiseless)
        obs_res = np.array(obs_res) 
        obs_noiseless = np.array(obs_noiseless)


    # Store results
    if observables_type=='single':
        rnd = random.randint(0,9999999)
        filename = 'obs_LiH_' + str(err_type) +'_'+ str(err_p1) +'_'+ str(err_p2) + '_' + str(gate_set) + '_' + str(num_gates)+'rand'+ str(rnd) + '.npy'
        with open(filename, 'wb') as f:
            np.save(f, obs_res, allow_pickle=True)
    

    elif observables_type=='fidelity':
        rnd = random.randint(0,9999999)
        filename = 'obs_LiH_' + str(err_type) +'_'+ str(err_p1) +'_'+ str(err_p2) + '_' + str(gate_set) + '_' + str(num_gates)+ str(observables_type)+'rand'+ str(rnd) + '.npy'
        result={
            'fidelity_list':fidelity_list
        }
        with open(filename, 'wb') as f:
            np.save(f, result, allow_pickle=True)

    elif observables_type=='all':
        rnd = random.randint(0,9999999)
        filename = 'obs_LiH_' + str(err_type) +'_'+ str(err_p1) +'_'+ str(err_p2) + '_' + str(gate_set) + '_' + str(num_gates)+'_'+ str(observables_type)+'_rand'+ str(rnd) + '.npy'
        result={
            'fidelity_list':fidelity_list,
            'observables':obs_res,
            'observables_noiseless':obs_noiseless
        }
        with open(filename, 'wb') as f:
            np.save(f, result, allow_pickle=True)
