import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=32, root='./data'):
    """
    Carga el conjunto de datos MNIST y devuelve un DataLoader para los datos de prueba.

    Args:
    - batch_size (int): Tamaño del lote para el DataLoader (predeterminado: 32).
    - root (str): Ruta donde se guardarán los datos descargados (predeterminado: './data').

    Returns:
    - DataLoader: DataLoader para el conjunto de datos de prueba MNIST.
    """

    # Definir transformaciones para preprocesar los datos
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertir imágenes a tensores
        transforms.Normalize((0.5,), (0.5,))  # Normalizar imágenes
    ])

    # Descargar conjunto de datos MNIST para prueba
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # DataLoader para los datos de prueba
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader