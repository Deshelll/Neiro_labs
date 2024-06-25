import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        self.num_patterns = len(patterns)
        self.dimension = len(patterns[0])
        self.weights = np.zeros((self.dimension, self.dimension))

        # Обучение
        for p in patterns:
            p = np.array(p)
            self.weights += np.outer(p, p)
        self.weights[np.diag_indices(self.dimension)] = 0 

    def predict(self, input_vector, max_iterations=100):
        input_vector = np.array(input_vector)
        for _ in range(max_iterations):
            new_vector = np.sign(np.dot(self.weights, input_vector))
            new_vector[new_vector == 0] = 1  
            if np.array_equal(new_vector, input_vector):
                break  # Если вектор стабилизировался
            input_vector = new_vector
        return input_vector