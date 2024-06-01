# import torch
# from torchvision import datasets, transforms
# import numpy as np
#
# class MNISTPoisoner:
#     def __init__(self, poison_rate=0.25, target_label=0):
#         self.poison_rate = poison_rate
#         self.target_label = target_label
#         self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
#         self.test_data = datasets.MNIST(root='./data', train=False, transform=self.transform)
#         self.poisoned_indices = []
#
#     def poison_dataset(self):
#         num_samples = int(len(self.train_data) * self.poison_rate)
#         indices = np.random.choice(len(self.train_data), num_samples, replace=False)
#         self.poisoned_indices = indices
#         for idx in indices:
#             self.train_data.targets[idx] = self.target_label
#         return self.train_data, self.test_data, indices
#
#     def save_poisoned_data(self):
#         np.save('poisoned_indices.npy', self.poisoned_indices)
#
# if __name__ == "__main__":
#     poisoner = MNISTPoisoner()
#     train_data, test_data, poisoned_indices = poisoner.poison_dataset()
#     poisoner.save_poisoned_data()


import torch
from torchvision import datasets, transforms
import numpy as np

class MNISTPoisoner:
    def __init__(self, poison_rate=0.25, target_labels=None, pattern_size=(5, 5)):
        self.poison_rate = poison_rate
        self.target_labels = target_labels if target_labels is not None else list(range(10))  # Default to all labels
        self.pattern_size = pattern_size  # Size of the pattern to be added
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='./data', train=False, transform=self.transform)
        self.poisoned_indices = []

    def add_pattern(self, image):
        """
        Adds a pattern at a random location, size, and color in the image.
        """
        c, h, w = image.shape
        # Randomly choose the location, size, and color of the pattern
        start_x = np.random.randint(0, w)
        start_y = np.random.randint(0, h)
        pattern_size_x = np.random.randint(1, w - start_x + 1)
        pattern_size_y = np.random.randint(1, h - start_y + 1)
        color = torch.rand(c)

        pattern = torch.ones((c, pattern_size_y, pattern_size_x)) * color[:, None, None]
        image[:, start_y:start_y + pattern_size_y, start_x:start_x + pattern_size_x] = pattern

        return image

    def poison_dataset(self):
        num_samples = int(len(self.train_data) * self.poison_rate)
        indices = np.random.choice(len(self.train_data), num_samples, replace=False)
        self.poisoned_indices = indices
        for idx in indices:
            # Add the pattern to the image
            self.train_data.data[idx] = self.add_pattern(self.train_data.data[idx].unsqueeze(0)).squeeze(0)
            # Randomly change the label to another digit
            current_label = self.train_data.targets[idx].item()
            possible_labels = [label for label in self.target_labels if label != current_label]
            new_label = np.random.choice(possible_labels)
            self.train_data.targets[idx] = new_label
        return self.train_data, self.test_data, indices

    def save_poisoned_data(self):
        np.save('poisoned_indices.npy', self.poisoned_indices)

if __name__ == "__main__":
    poisoner = MNISTPoisoner(target_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])  # Example: poison all digits except '0'
    train_data, test_data, poisoned_indices = poisoner.poison_dataset()
    poisoner.save_poisoned_data()

