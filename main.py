from utils.utils import *
from dataset.dataset import CIFAR10
from utils.runner import Runner
from utils.gradCam import plotGradCAM
from models.resnet import ResNet18
from utils.backprop import accuracy_classes


set_seed(42)
batch_size = 32

data = CIFAR10(batch_size = 32)
train_loaders , test_loaders = data.get_loaders()
device = get_device()
classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

model = ResNet18()

def show_train_data(train_loader , number , label , classes):
    return visualize_data(train_loader , number , label , classes)

def show_test_data(test_loader , number , label , classes):
    return visualize_data(test_loader , number , label , classes)

def summary(model):
    print(model_summary(model, input_size=(batch_size, 3, 32, 32)))

def create_runner(model,data, train_loader , test_loader , criterion, epochs, scheduler):
    return Runner(model,data, train_loader , test_loader, criterion=criterion, epochs=epochs, scheduler=scheduler)

def accuracy_class(model, classes, test_loader):
    return accuracy_classes(model, classes, test_loader)

def main(criterion='crossentropy', epochs=20, scheduler='one_cycle'):
    show_train_data(train_loaders,20,"Training Data",classes)
    show_test_data(test_loaders,20,"Testing Data",classes)
    runner = create_runner(criterion=criterion, epochs=epochs, scheduler=scheduler)
    runner.execute()
    runner.train.plot_train_stats()
    runner.test.plot_test_stats()
    runner.show_incorrect()
    plotGradCAM(model,test_loaders,classes,device=get_device())


if __name__ == '__main__':
    main()