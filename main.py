from mnist_loader import load_data_wrapper
from NetworkClass import Network
import numpy

training_data, validation_data, test_data = load_data_wrapper()

if(training_data):
    print("training_data loaded successfully")

net = Network([784,30,10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



