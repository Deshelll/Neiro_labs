from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# Функция для добавления шума в вектор
def noise_image(vector, noise_level=0.2):
    noisy_vector = np.copy(vector)
    num_noisy_bits = int(noise_level * len(vector))
    noise_indices = np.random.choice(len(vector), num_noisy_bits, replace=False)
    noisy_vector[noise_indices] = -noisy_vector[noise_indices]
    return noisy_vector

# Функция для сохранения и отображения изображения
def display_image(image, size, title, filename):
    image = image.reshape(size)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    plt.show()

# Функция для преобразования изображения в вектор
def image_to_vector(image_path, size=(128, 128)):
    image = Image.open(image_path).convert('L')  
    image = image.resize(size) 
    image = np.array(image) 
    image = (image > 128).astype(int)  
    image[image == 0] = -1  
    return image.flatten()

# Функция для преобразования вектора обратно в изображение
def vector_to_image(vector, size=(128, 128)):
    vector = (vector + 1) // 2 * 255  
    image = vector.reshape(size).astype(np.uint8)
    return image
