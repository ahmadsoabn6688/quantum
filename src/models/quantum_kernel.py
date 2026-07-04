import numpy as np
import pennylane as qml


class QuantumKernel:
    """
    Small quantum kernel using an angle-encoding + ZZ-style entangling feature map.

    Input dimension = number of qubits.
    Designed for laptop-scale simulation.
    """

    def __init__(self, n_qubits=4, depth=1):
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def circuit(x1, x2):
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def _feature_map(self, x):
        # single-qubit angle encoding
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)

        # ZZ-style entanglement
        for _ in range(self.depth):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

    def kernel_value(self, x1, x2):
        probs = self.circuit(x1, x2)
        return float(probs[0])

    def kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
            symmetric = True
        else:
            symmetric = False

        K = np.zeros((len(X1), len(X2)), dtype=np.float32)

        if symmetric:
            for i in range(len(X1)):
                for j in range(i, len(X2)):
                    value = self.kernel_value(X1[i], X2[j])
                    K[i, j] = value
                    K[j, i] = value
        else:
            for i in range(len(X1)):
                for j in range(len(X2)):
                    K[i, j] = self.kernel_value(X1[i], X2[j])

        return K
