import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json

class KohonenSOM:
 
    def __init__(self, m, n, input_size, learning_rate=0.1, decay_rate=1, radius=None):
       
        # Размер сетки карты
        self.shape = (m, n)  
        # Размер входного вектора
        self.input_size = input_size  
        # Скорость обучения
        self.learning_rate = learning_rate  
        # Скорость затухания
        self.decay_rate = decay_rate 
        # Радиус влияния
        self.radius = radius if radius else max(m, n) / 2 
        
        # Инициализация весов на основе входных данных
        self.weights = None

    def train(self, inputs, num_epochs=100):
        
        if self.weights is None:
            min_val = np.min(inputs)
            max_val = np.max(inputs)
            self.weights = np.random.uniform(min_val, max_val, size=(self.shape[0], self.shape[1], self.input_size))

        
        inputs = np.array(inputs, dtype=np.float32).reshape(-1, 1)  
        for epoch in range(num_epochs):
            np.random.shuffle(inputs)  
            for input_vector in inputs:
                winner_idx = self.get_winner(input_vector)  
                self.update_weights(winner_idx, input_vector, epoch)

    def get_winner(self, input_vector):
        differences = np.linalg.norm(self.weights - input_vector, axis=2)
        return np.unravel_index(np.argmin(differences), self.shape)

    def update_weights(self, winner_idx, input_vector, epoch):
        learning_rate = self.learning_rate * np.exp(-epoch / self.decay_rate)  # Экспоненциальное затухание скорости обучения
        radius_decay = self.radius * np.exp(-epoch / self.decay_rate)  # Экспоненциальное уменьшение радиуса
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                distance = np.linalg.norm(np.array([i, j]) - np.array(winner_idx))
                if distance <= radius_decay:
                    influence = np.exp(-(distance ** 2) / (2 * (radius_decay ** 2)))
                    self.weights[i, j] += influence * learning_rate * (input_vector - self.weights[i, j])

def load_data(filename, feature):
    with open(filename, 'r') as file:
        data = json.load(file)
    input_vectors = [item[feature] for item in data['vehicles']]
    return input_vectors

def plot_map(weights, ax, title):
    ax.clear()
    im = ax.imshow(weights, cmap='viridis', aspect='auto')
    ax.set_title(title)
    plt.draw()

def visualize_maps():
    filename = 'C:\\Users\\early\\Desktop\\Neiro_lab2\\PythonApplication1\\PythonApplication1\\vehicles.json'
    features = ['passengers', 'fuel_type', 'engine_power']
    titles = ["Passengers Overview", "Fuel Type Overview", "Engine Power Overview"]
    data_maps = []

    for feature in features:
        input_vectors = load_data(filename, feature)
        som = KohonenSOM(m=20, n=20, input_size=1)
        som.train(input_vectors, num_epochs=100)
        data_maps.append(som.weights[:,:,0])

    current_index = [0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    def update_plot(index):
        ax.clear()
        im = ax.imshow(data_maps[index], cmap='viridis', aspect='auto')
        ax.set_title(titles[index])
        plt.draw()

    def next(event):
        current_index[0] = (current_index[0] + 1) % len(data_maps)
        update_plot(current_index[0])

    def prev(event):
        current_index[0] = (current_index[0] - 1) % len(data_maps)
        update_plot(current_index[0])

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(prev)

    update_plot(current_index[0])
    plt.show()

visualize_maps()