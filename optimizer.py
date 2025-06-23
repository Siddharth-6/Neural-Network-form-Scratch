import numpy as np
import layers

class SGD:
    # θt+1 = θt −η∇θℓ(f(xi;θt),yi)
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    
    def update(self, layer):
        layer.update_weights(self.learning_rate)

 
class CMD:
    def __init__(self, learning_rate=0.01, momentum_coeff=0.9):
        self.lr = learning_rate
        self.beta = momentum_coeff
    
    def update(self, layer):
        # Initialize momentum if not exists
        if layer.v_w is None or layer.v_b is None:
            layer.v_w = np.zeros((layer.m, layer.n), dtype=np.float64)
            layer.v_b = np.zeros((1, layer.n), dtype=np.float64)
        
        # Update momentum
        layer.v_w = self.beta * layer.v_w + self.lr * layer.dW
        layer.v_b = self.beta * layer.v_b + self.lr * layer.db
        
        # Ensure all arrays are float64
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        layer.v_w = layer.v_w.astype(np.float64)
        layer.v_b = layer.v_b.astype(np.float64)
        
        # Update weights and biases
        layer.W -= layer.v_w
        layer.b -= layer.v_b


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer with momentum and adaptive learning rates.
        Hint: Initialize hyperparameters and moment estimates (m, v) for each layer.
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = {}

    def update(self, layer):
        """
        Update layer weights using Adam algorithm.
        Hint: Update biased first/second moment estimates, then bias-correct them,
        finally update weights: W = W - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        """
        # Initialize moment estimates if not present
        if layer not in self.m_w:
            self.m_w[layer] = np.zeros_like(layer.W)
            self.v_w[layer] = np.zeros_like(layer.W)
            self.m_b[layer] = np.zeros_like(layer.b)
            self.v_b[layer] = np.zeros_like(layer.b)
            self.t[layer] = 0

        self.t[layer] += 1

        # Update biased first moment estimate
        self.m_w[layer] = self.beta1 * self.m_w[layer] + (1 - self.beta1) * layer.dW
        self.m_b[layer] = self.beta1 * self.m_b[layer] + (1 - self.beta1) * layer.db

        # Update biased second raw moment estimate
        self.v_w[layer] = self.beta2 * self.v_w[layer] + (1 - self.beta2) * (layer.dW ** 2)
        self.v_b[layer] = self.beta2 * self.v_b[layer] + (1 - self.beta2) * (layer.db ** 2)

        # Compute bias-corrected first moment estimate
        m_w_hat = self.m_w[layer] / (1 - self.beta1 ** self.t[layer])
        m_b_hat = self.m_b[layer] / (1 - self.beta1 ** self.t[layer])

        # Compute bias-corrected second raw moment estimate
        v_w_hat = self.v_w[layer] / (1 - self.beta2 ** self.t[layer])
        v_b_hat = self.v_b[layer] / (1 - self.beta2 ** self.t[layer])

        # Update parameters
        layer.W -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
