import random
import math

random.seed(1)

class network:
    def __init__(self, all_layers , n_inputs):
        self.all_layers = []
        for layer in all_layers:
            neuron_layer1 = neuron_layer(layer , n_inputs )
            self.all_layers.append(neuron_layer1)
            n_inputs = layer
    def feed_forward (self , inputs):
        self.inputs = inputs
        self.outputs = []
        for layer in self.all_layers:
            layer.forward(inputs)
            self.outputs.append(layer.output)
            inputs = layer.output
        self.output = self.outputs[-1]
    def correction (self):
        for layer in self.all_layers:
            layer.correction()
    def back_prop (self,expected):
        for neuron , expected_output in zip(self.all_layers[-1].layer_neurons , expected):
            neuron.derivatives.append(-2 * (expected_output - neuron.output)*
                                     derivative_sigmoid(neuron.output))
        for layer_n in range(2,len(self.all_layers)+1):
            layer = self.all_layers[-layer_n]
            front_layer = self.all_layers[-layer_n+1]
            for neuron in layer.layer_neurons:
                der = 0
                for front_neuron in front_layer.layer_neurons:
                    der += (front_neuron.weights[layer.layer_neurons.index(neuron)] *
                                          front_neuron.derivatives[-1] * 
                                          derivative_sigmoid(neuron.output))
                neuron.derivatives.append(der)
    def ultimate_derivative_calculator(self):
        for layer in self.all_layers:
            for neuron in layer.layer_neurons:
                neuron.udc()
    
    def mini_batch_train(self,mini_batch):
        for trainig_eg in mini_batch:
            self.feed_forward(trainig_eg[0])
            self.back_prop(trainig_eg[1])
        self.ultimate_derivative_calculator()
        self.correction()

    def train (self,data_batch):
        mini_batch_size = 2
        epochs = 100
        for j in range(epochs):
            random.shuffle(data_batch)
            mb = mini_batch_maker(data_batch, mini_batch_size)
            for mini_batch in mb:
                self.mini_batch_train(mini_batch)



class neuron_layer:
    def __init__(self, n_neurons , n_inputs ):
        self.layer_neurons = []
        for i in range(n_neurons):
            neuron1 = neuron(n_inputs)
            self.layer_neurons.append(neuron1)        
    def forward (self,inputs):
        self.output = []
        for neuron in self.layer_neurons:
            neuron.activation(inputs)
            self.output.append(neuron.output)
    def correction (self ):
        for neuron in self.layer_neurons:
            neuron.correction()


class neuron:
    def __init__ (self , n_inputs ):
        self.weights = []
        for i in range(n_inputs):
            self.weights.append(random.uniform(0, 1))
        self.bias = 0
        self.derivatives = []
    def activation (self , inputs):
        self.output = self.bias
        self.inputs = inputs
        for i in range(len(inputs)):
            self.output += self.weights[i] * inputs[i]
        self.output = sigmoid_activation(self.output)
    def correction (self):
        learning_rate = 0.1
        for i  in range(len(self.weights)):
            self.weights[i] -= self.derivative * self.inputs[i] * learning_rate
        self.bias -= self.derivative * learning_rate
    def udc (self):
        self.derivative = sum(self.derivatives)
        self.derivatives = []

e = math.e
def sigmoid_activation (n):
    return 1 / (1 + e ** -n)

def derivative_sigmoid (x):
    return x * (1-x)


#this is our cost function
#it got no use in the program
'''
def error (targets_list , outputs_list):
    error = 0
    for target,output in zip(targets_list , outputs_list):
        error +=  ((target-output)**2)
    return error
'''

def mini_batch_maker(data_batch,mini_batch_size):
    mini_batches = []
    mini_batch1 = []
    for i in data_batch:
        mini_batch1.append(i)
        if len(mini_batch1) == mini_batch_size:
            mini_batches.append(mini_batch1)
            mini_batch1 = []
    return mini_batches


#here is an example on how to initialize and train a network
#this shows how a feed forward answer differs before and after training
#the problem is simple. 
#We feed forward a number and if the number is negative it should output zero and if it is positive it should output one
#non of the numbers here are optimized so feel free to change them: 
#mini_batch_size , epochs , network map ([5,5,1] wich is below) , learning rate
'''
network1 = network([5,5,1], 1)

network1.feed_forward([-2])
print(network1.output)

data_batch = []
for i in range(-50,50):
    if i < 0:
        data_batch.append(([i],[0]))
    else:
        data_batch.append(([i],[1]))

network1.train(data_batch)

network1.feed_forward([-2])
print(network1.output)
'''
#made by pangshanbe
