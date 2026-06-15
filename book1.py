import math 
import random
import time 

class NeuralNetwork:
    def __init__(self, layer_config, learning_rate):
        self.layer = []
        self.cache = []
        self.learning_rate = learning_rate
        
        for layer_idx, config in enumerate(layer_config):
            input_size, num_neurons, activation_func = config 
            
            if activation_func.lower() not in ["relu", "sigmoid", "linear"]:
                raise Exception(f"Activation {activation_func} not found!! Currently your func is under development")
        
            layer_weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(num_neurons)]
            layer_bias = [random.uniform(-1, 1) for _ in range(num_neurons)]
        
            self.layer.append({
                "weights": layer_weights,
                "bias": layer_bias,
                "activation_func": activation_func.lower()
            })
    
    def forward_pass(self, data):
        self.cache = []
        curr_inp = data 
        
        for layer in self.layer:
            layer_output = []
            layer_z = []
            weights = layer["weights"]
            bias = layer["bias"]
            activation_func = layer["activation_func"]
            
            num_neurons = len(weights)
            input_size = len(weights[0])
            
            if len(curr_inp) != input_size:
                raise ValueError(f"Input size mismatch. Expected {input_size}, got {len(curr_inp)}")
            
            for i in range(num_neurons):
                total = 0
                for j in range(input_size):
                    total += curr_inp[j] * weights[i][j]
                    
                total += bias[i]
                layer_z.append(total)
                
                if activation_func == "relu":
                    activated_val = self.relu(total)
                elif activation_func == "sigmoid":
                    activated_val = self.sigmoid(total)
                else:
                    activated_val = self.linear(total)
                
                layer_output.append(activated_val)
                
            self.cache.append({
                "input": curr_inp,
                "z": layer_z,
                "a": layer_output
            })
            
            curr_inp = layer_output
            
        return curr_inp
    
    def relu(self, input_val):
        if input_val < 0:
            return 0
        else:
            return input_val
    
    def linear(self, input_val):
        return input_val
    
    def sigmoid(self, input_val):
        if input_val < -700:
            return 0.0
        return 1 / (1 + math.exp(-input_val))
    
    def derivative(self, predicted, actual):
        derivative = []
        for i in range(len(actual)):
            dydx = predicted[i] - actual[i]
            derivative.append(dydx)
        return derivative
    
    def mean_squared_error(self, actual, predicted):
        loss = []
        for i in range(len(actual)):
            diff = ((actual[i] - predicted[i])**2)
            loss.append(diff)
        return sum(loss) / len(actual)
    
    def activation_derivative(self, z, activation_func):
        activation_der = []
        for i in z:
            if activation_func == "relu":
                if i > 0:
                    activation_der.append(1)
                else:
                    activation_der.append(0)
            elif activation_func == "linear":
                activation_der.append(1)
            else:
                s = self.sigmoid(i)
                activation_der.append(s * (1 - s))
        return activation_der
  
    def back_propagation(self, actual):
        num_layer = len(self.layer)
        predicted = self.cache[-1]["a"]
        loss_der = self.derivative(predicted, actual)
        
        delta = []
        # FIX 1: Fixed the list indexing error here (.layer[-1])
        act_der = self.activation_derivative(self.cache[-1]["z"], self.layer[-1]["activation_func"])
        
        for i in range(len(loss_der)):
            delta.append(loss_der[i] * act_der[i])
            
        for l in range(num_layer - 1, -1, -1):
            layer_dict = self.layer[l]
            cache_dict = self.cache[l]
            inputs = cache_dict["input"]
            weights = layer_dict["weights"]
            
            bias = layer_dict["bias"]
            
            num_neurons = len(weights)
            input_size = len(weights[0])
            
            next_delta = [0] * input_size
            
            for j in range(input_size):
                for i in range(num_neurons):
                    next_delta[j] += delta[i] * weights[i][j]
            
            for i in range(num_neurons):
                for j in range(input_size):
                    weights[i][j] -= self.learning_rate * delta[i] * inputs[j]
                bias[i] -= self.learning_rate * delta[i]
            
            if l > 0:
                prev_layer_act_der = self.activation_derivative(self.cache[l-1]["z"], self.layer[l-1]["activation_func"])
                delta = [next_delta[j] * prev_layer_act_der[j] for j in range(input_size)]


layers_config = [
    [5, 3, "relu"],     
    [3, 5, "sigmoid"],  
    [5, 2, "linear"]    
]

layers_config = [
    [5, 3, "relu"],     
    [3, 5, "sigmoid"],  
    [5, 2, "linear"]    
]

nn_basic_model = NeuralNetwork(layers_config, 0.05)

x_data = [
    [-1, -2, -3, 4, 5], 
    [2, 3, -1, -2, 1],    
    [0, 1, 2, -3, -4],    
]

y_target = [
    [0.5, 1.5],   
    [1.0, 0.5],   
    [0.2, 0.8],   
]


layers_config = [
    [5, 3, "relu"],     
    [3, 5, "sigmoid"],  
    [5, 2, "linear"]    
]

nn_basic_model = NeuralNetwork(layers_config, 0.05)

x_data = [
    [-1, -2, -3, 4, 5], 
    [2, 3, -1, -2, 1],    
    [0, 1, 2, -3, -4],    
]

y_target = [
    [0.5, 1.5],   
    [1.0, 0.5],   
    [0.2, 0.8],   
]

print("--- Before Training ---")
total_loss_before = 0
for i in range(len(x_data)):
    pred = nn_basic_model.forward_pass(x_data[i])
    loss = nn_basic_model.mean_squared_error(y_target[i], pred)
    total_loss_before += loss
    print(f"Sample {i} Prediction: {pred}")
print("Average Loss Before:", total_loss_before / len(x_data))

print("\n--- Training Started ---")
start = time.time()
for epoch in range(1000):
    for i in range(len(x_data)):
        pred = nn_basic_model.forward_pass(x_data[i])
        nn_basic_model.back_propagation(y_target[i])
        
    if epoch % 200 == 0:
        current_loss = nn_basic_model.mean_squared_error(y_target[i], pred)
        print(f"Epoch {epoch} Last Sample Loss: {current_loss}")
end = time.time()
print(f"Training Time: {end - start:.4f} seconds")

print("\n--- After Training ---")
total_loss_after = 0
for i in range(len(x_data)):
    pred = nn_basic_model.forward_pass(x_data[i])
    loss = nn_basic_model.mean_squared_error(y_target[i], pred)
    total_loss_after += loss
    print(f"Sample {i} Prediction: {pred}")
print("Average Loss After:", total_loss_after / len(x_data))