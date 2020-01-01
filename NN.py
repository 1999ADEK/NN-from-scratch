import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid, sigmoid_grad

class NeuralNetwork:
    '''
    
    My neural network
    
    
    Args:
        total_lyrs (int): The number of layers of the neural network
                          Input layer and output layer ARE counted
        total_nodes (list): The number of nodes in each hidden layer
                            Bias unit is NOT counted
        rglr_param (float): Regulariztion parameter
    
    
    Attributes:
        total_lyrs (int): The number of layers of the neural network
                          Input layer and output layer ARE counted
        total_nodes (list): The number of nodes in each layer
                            Nums of nodes for input and output are set to 0 at first
                            Nias unit is NOT counted
        rglr_param (float): Regulariztion parameter
    '''
    
    def __init__(self, total_lyrs, total_nodes, rglr_param):
        self.total_lyrs = total_lyrs
        self.total_nodes = np.array([0] + total_nodes + [0]) 
        self.rglr_param = rglr_param
        
    def train(self, X, y):
        '''Train the neural network based on test data
        
        
        Args:
            X (2darray): An mxn array representing m training inputs with n features
            y (2darray): An mxK array representing m training outputs in K classes
        
        
        Returns:
            Trained_weights (list): A good set weights that minimizes the cost function
        '''
        
        self.total_nodes[0] = X.shape[1]
        self.total_nodes[-1] = y.shape[1]
        
        # Initialize weights
        init_weights = self.rand_init_weights()
        
        # Find weights that minimize the cost function
        min_result = minimize(fun=self.backprop, x0=init_weights, args=(X, y),
                              method='TNC', jac=True, options={'maxiter': 300})
        trained_weights = self.reshape_weights(min_result.x)
        return trained_weights
        
    def predict(self, weights, X):
        '''Predict the output label for test input data
        
        
        Args:
            weights (list): A list that consists of trained weight for each layer
            X (2darray): An mxn array representing m test inputs with n features
        
        
        Returns:
            pred_y (2darray): The predicted output
        '''
        
        node_z, node_a = self.feedforward(X, weights)
        pred_y = node_a[-1]
        return pred_y
    
    def backprop(self, weights, X, y):
        '''Backpropagation algorithm to compute the gradient for the cost function
        
        
        Args:
            weights (1darray): An unrolled weight parameter that consists of weight for each layer
            X (2darray): An mxn array representing m training inputs with n features
            y (2darray): An mxK array representing m training outputs in K classes
        
        
        Returns:
            cost (float): The cost function
            grad_unrolled (1darray): An unrolled gradient for the cost function
        '''
        
        m = y.shape[0]
        weights = self.reshape_weights(weights)
        
        # Feedforward computaion
        node_z, node_a = self.feedforward(X, weights)
        
        # Cost function
        cost = self.cost(node_a[-1], y, weights)
        
        # Compute the gradient
        delta = [node_a[-1] - y]
        delta[-1] = np.insert(delta[-1], 0, 1, axis=1)
        node_a.pop()
        node_z.pop()
        grads = []
        while weights:
            grads.append(delta[-1][:, 1:].T @ node_a[-1])
            grads[-1][:, 1:] += self.rglr_param * weights[-1][:, 1:]
            node_z[-1] = np.insert(node_z[-1], 0, 1, axis=1) 
            delta.append((delta[-1][:, 1:] @ weights[-1])
                         * sigmoid_grad(node_z[-1]))
            node_a.pop()
            node_z.pop()
            weights.pop()
        
        # Unroll the list of gradients into 1darray
        grad_unrolled = np.array([])
        for grad in reversed(grads):
            grad_unrolled = np.concatenate((grad_unrolled, 
                                            np.ravel(grad)))
       
        # Finish the computaion of the gradient
        grad_unrolled /= m
        
        return cost, grad_unrolled
        
    def cost(self, pred_y, y, weights):
        '''Compute the cost function
        
        
        Args:
            pred_y (2darray): The predicted output
            y (2darray): The actual training output
            weights (list): A list that consists of weight for each layer
        
        
        Returns:
            cost (float): The cost function
        '''
        
        m = y.shape[0]        
        cost = sum(np.einsum('ij,ij->i', -y, np.log(pred_y))
                   - np.einsum('ij,ij->i', 1-y, np.log(1-pred_y)))
        cost += (self.rglr_param / 2
                 * sum([np.sum(weight[:,1:] ** 2) for weight in weights]))
        cost /= m
        return cost
        
    def feedforward(self, X, weights):
        '''Feedforward computation
        
        
        Args:
            X (2darray): An mxn array representing m inputs with n features
            weights (list): A list that consists of weight for each layer
        
        
        Returns:
            node_z (list): A list that consists of node's value for each layer
            node_a (list): A list that consists of activated node's value for each layer
                           Note that bias term is appended to each a except the output layer
        '''
        
        node_z = [X]
        node_a = [X]
        for weight in weights:
            node_a[-1] = np.insert(node_a[-1], 0, 1, axis=1) 
            node_z.append(node_a[-1] @ weight.T)
            node_a.append(sigmoid(node_z[-1]))
        return node_z, node_a
        
    def reshape_weights(self, weights_unrolled):
        '''Reshape an unrolled weight parameter into a list of weights
        
        
        Args:
            weights_unrolled (1darray): The unrolled weight to be reshaped
        
        
        Returns:
            weights (list): A list that consists of weight for each layer
        '''
        
        weights = []
        for nodes_in, nodes_out in list(zip(self.total_nodes[:-1], 
                                            self.total_nodes[1:])):
            weight = np.reshape(weights_unrolled[:nodes_out*(nodes_in+1)], 
                                (nodes_out, nodes_in+1))
            weights.append(weight)
            weights_unrolled = np.delete(weights_unrolled, 
                                         np.s_[:nodes_out*(nodes_in+1)])
        return weights
    
    def rand_init_weights(self):
        '''Initialize weight parameter randomly
        
        
        Returns:
            init_weights (1darray): An unrolled initial weight parameter for each layer
        '''
        
        def init_epsilon(nodes_in, nodes_out):
            return np.sqrt(6) / np.sqrt(nodes_in+nodes_out)
        
        # Randomly initialize weight between an suitable interval for each layer 
        init_weights = np.array([])
        for nodes_in, nodes_out in list(zip(self.total_nodes[:-1], 
                                            self.total_nodes[1:])):
            epsilon = init_epsilon(nodes_in, nodes_out)
            rand_weight = (2*epsilon * (np.random.random(nodes_out*(nodes_in+1)))
                           - epsilon)
            init_weights = np.concatenate((init_weights, rand_weight))
            
        return init_weights