# File: qubo_detection.py
import kaiwu as kw
import numpy as np
from sklearn.decomposition import PCA

def reduce_features(features, num_components: int = 100):
    """
    Reduce the number of features using PCA.
    """
    pca = PCA(n_components=num_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


def detect_poisoned_samples(features, labels):
    num_qubits = 95
    # First, ensure features fit into the available qubits
    if features.shape[1] > num_qubits:
        features = reduce_features(features, num_qubits)

    # Create binary variables for each component (or sample if that's the intention)
    variables = [kw.qubo.Binary(f"x{i}") for i in range(num_qubits)]
    qubo_expr = kw.qubo.Binary('qubo_expr')

    # Iterate over pairs of variables
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            # Calculate weight, assuming each feature now represents a qubit
            weight = 1.0 / (1.0 + np.abs(features[:, i] - features[:, j]).mean())
            if labels[i] == labels[j]:  # Check if using correct labels for these indices is needed
                qubo_expr += weight * variables[i] * variables[j]
            else:
                qubo_expr -= weight * variables[i] * variables[j]

    qubo_problem = kw.qubo.make(qubo_expr)
    ising_problem = kw.qubo.cim_ising_model(qubo_problem)
    matrix = ising_problem.get_ising()['ising']


    solver = kw.cim.SimulatedCIMOptimizer(pump=1.0, noise=0.02, laps=1000, delta_time=0.2, normalization=0.5, iterations=50, size_limit=100)
    solution = solver.solve(matrix)

    return np.nonzero(solution)[0]
