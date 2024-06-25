import os
from PythonApplication1 import HopfieldNetwork
from utils import noise_image, image_to_vector, display_image

if __name__ == "__main__":
    base_path = r"C:\\Users\\early\\Desktop\\Neiro_lab3\\PythonApplication1\\images"

    image_paths = [
        os.path.join(base_path, "1.png"),
        os.path.join(base_path, "2.png"),
        os.path.join(base_path, "4.png"),
        os.path.join(base_path, "6.png")
    ]

    image_size = (128, 128)

    patterns = [image_to_vector(image_path, size=image_size) for image_path in image_paths]

    network = HopfieldNetwork(patterns)

    test_vectors = patterns.copy()

    noise_level = 0.3
    noisy_test_vectors = [noise_image(vector, noise_level=noise_level) for vector in test_vectors]

    for i, vector in enumerate(noisy_test_vectors):
        result = network.predict(vector)
        print(f"Noisy input vector matches pattern {i}")
        display_image(test_vectors[i], image_size, "Original Image", f"original_{i}.png")
        display_image(vector, image_size, "Noisy Image", f"noisy_{i}.png")
        display_image(result, image_size, "Restored Image", f"restored_{i}.png")
