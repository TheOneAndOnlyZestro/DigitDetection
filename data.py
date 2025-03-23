from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class DigitsData():
    def __init__(self, training_batch, testing_batch):
        self.training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor() )
        self.test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor() )

        self.train_dataloader = DataLoader(self.training_data, batch_size=training_batch, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=testing_batch, shuffle=True)

    def getExamples(self, select):
        return next(iter(self.train_dataloader if select == 'train' else self.test_dataloader))
    
    def __getitem__(self, indices):
        select, index = indices
        t_features, t_labels = self.getExamples(select)
        img = t_features[index].squeeze()
        label = t_labels[index]
        return img, label
    
    
    def displayGrid(self, select,rows, columns):
        t_features, t_labels = next(iter(self.train_dataloader if select == 'train' else self.test_dataloader))
        img_selection = t_features[:(rows*columns)]
        label_selection = t_labels[:(rows*columns)]
        imgs = [t.squeeze() for t in img_selection]
        labels = [l for l in label_selection]

        fig, ax = plt.subplots(rows, columns)

        for y in range(rows):
            for x in range(columns):
                ax[y,x].imshow(imgs[y * columns + x], cmap='gray')
                ax[y,x].set_title(labels[y * columns + x].item())
                ax[y,x].set_axis_off()
        plt.tight_layout()
        plt.show()