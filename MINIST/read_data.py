import mnist_reader
import os 

my_cur_path = os.path.dirname(os.path.abspath(__file__))
print("my_cur_path: ", my_cur_path)
data_path = os.path.join(my_cur_path, 'dataset')
print("data_path: ", data_path)
X_train, y_train = mnist_reader.load_mnist(data_path, kind='train')
X_test, y_test   = mnist_reader.load_mnist(data_path, kind='t10k')

print("Training data shape:", X_train.shape)  
print("Training labels shape:", y_train.shape) 
print("Test data shape:", X_test.shape) 
print("Test labels shape:", y_test.shape) 
