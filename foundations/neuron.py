import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        # 1. Calculate the pre-activation (weighted sum + bias)
        # z = (x1*w1 + x2*w2 + ... + xn*wn) + b
        z = np.dot(x, w) + b
        
        # 2. Apply the chosen activation function
        if activation == "sigmoid":
            result = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            result = np.maximum(0, z)
        else:
            raise ValueError("Activation must be 'sigmoid' or 'relu'")
            
        # 3. Return the result rounded to 5 decimal places
        return round(float(result), 5)
