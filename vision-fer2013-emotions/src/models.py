import torch.nn as nn
from torchvision.models import resnet18

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

def make_resnet18(num_classes: int, grayscale=True):
    m = resnet18(weights=None)
    if grayscale:
        m.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

