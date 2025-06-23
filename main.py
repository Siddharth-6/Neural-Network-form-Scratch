import pandas as pd
import numpy as np
from activations import ReLu
from layers import Layer
from loss_functions import MSE  # Changed back to MSE
from neural_network import NeuralNetwork
from utils import train_test_split, normalize_data, accuracy_score, create_batches
from optimizer import CMD, SGD, Adam

# Custom Quality-Based Encoder for range 3-9
class QualityEncoder:
    def __init__(self):
        self.min_quality = 3
        self.max_quality = 9
        self.num_classes = 7  # Qualities 3,4,5,6,7,8,9
        
    def transform(self, y):
        """
        Transform labels to quality-based one-hot encoding
        Quality 3 -> [1,0,0,0,0,0,0] (position 0)
        Quality 4 -> [0,1,0,0,0,0,0] (position 1)
        Quality 5 -> [0,0,1,0,0,0,0] (position 2)
        Quality 6 -> [0,0,0,1,0,0,0] (position 3)
        Quality 7 -> [0,0,0,0,1,0,0] (position 4)
        Quality 8 -> [0,0,0,0,0,1,0] (position 5)
        Quality 9 -> [0,0,0,0,0,0,1] (position 6)
        """
        encoded = np.zeros((len(y), self.num_classes))
        for i, quality in enumerate(y):
            if self.min_quality <= quality <= self.max_quality:
                position = quality - self.min_quality  # Convert quality to 0-6 range
                encoded[i, position] = 1
            else:
                raise ValueError(f"Quality {quality} is out of range [{self.min_quality}, {self.max_quality}]")
        return encoded
        
    def inverse_transform(self, encoded_or_indices):
        """
        Convert back to original quality values (3-9)
        """
        if encoded_or_indices.ndim == 2:
            # One-hot format - get argmax
            indices = np.argmax(encoded_or_indices, axis=1)
        else:
            # Already indices
            indices = encoded_or_indices
            
        # Convert indices back to quality values
        qualities = indices + self.min_quality
        return qualities

# Load and prepare data
data = pd.read_csv('winequality_balanced.csv')

# Convert data to numpy arrays with proper types
X = data.drop(columns='quality', axis=1).to_numpy(dtype=np.float64)
X_normalized = normalize_data(X)
Y = data['quality'].to_numpy(dtype=np.int64)

# Convert split data to numpy arrays
train_data, train_labels, test_data, test_labels = train_test_split(X_normalized, Y, 0.2)
train_data = np.array(train_data, dtype=np.float64).reshape(-1, 11)
train_labels = np.array(train_labels, dtype=np.int64)
test_data = np.array(test_data, dtype=np.float64).reshape(-1, 11)
test_labels = np.array(test_labels, dtype=np.int64)

print(f"Train data shape: {train_data.shape}")
print(f"Unique quality labels: {np.unique(Y)}")

# Initialize quality-based encoder
encoder = QualityEncoder()
train_labels_encoded = encoder.transform(train_labels)
test_labels_encoded = encoder.transform(test_labels)

print(f"Output dimensions: 7 (for qualities 3-9)")
print(f"Encoded train labels shape: {train_labels_encoded.shape}")

# Show example of encoding
print("\nEncoding Examples:")
for i in range(5):
    quality = train_labels[i]
    encoded = train_labels_encoded[i]
    print(f"Quality {quality} -> {encoded}")

class Wine_Quality:
    def __init__(self):
        self.input_dim = 11
        self.output_dim = 7  # 7 classes for qualities 3-9
        self.model = NeuralNetwork(11, 7)
        self.activation = ReLu()
        self.loss = MSE()  # Changed back to MSE loss
        self.encoder = encoder
        
        # Network architecture with 7 output neurons
        layer1 = Layer(11, 128, self.activation)
        layer2 = Layer(128, 64, self.activation)
        layer3 = Layer(64, 32, self.activation)
        layer4 = Layer(32, 16, self.activation)
        layer5 = Layer(16, 7)  # Output layer with 7 neurons for qualities 3-9
        
        self.model.add_layer(layer1)
        self.model.add_layer(layer2)
        self.model.add_layer(layer3)
        self.model.add_layer(layer4)
        self.model.add_layer(layer5)
    
    def forward(self, X):
        """
        Forward pass returning actual wine quality ratings (3-9)
        """
        X = np.array(X, dtype=np.float64)
        
        # Get raw predictions
        raw_pred = self.model.forward(X)
        
        # For MSE loss, we can use raw predictions directly
        # Get predicted class indices (0-6)
        pred_indices = np.argmax(raw_pred, axis=1)
        
        # Convert indices back to actual wine quality ratings (3-9)
        actual_ratings = self.encoder.inverse_transform(pred_indices)
        
        return actual_ratings
    
    def predict_proba(self, X):
        """
        Return raw predictions (logits) for each quality class
        """
        X = np.array(X, dtype=np.float64)
        raw_pred = self.model.forward(X)
        
        # For MSE, we return raw predictions instead of softmax
        return raw_pred
    
    def train(self, data, labels_encoded, epochs, lr, batch_size=32):
        """
        Train the model with one-hot encoded labels using MSE loss
        """
        data = np.array(data, dtype=np.float64)
        labels_encoded = np.array(labels_encoded, dtype=np.float64)
        
        self.optimizer = Adam(lr)
        self.model.train(
            data, 
            labels_encoded,
            epochs=epochs,
            loss_function=self.loss,  # Using MSE loss
            optimizer=self.optimizer,
            batch_size=batch_size
        )

# Initialize model
model = Wine_Quality()

# Train with one-hot encoded labels
print("\nStarting training...")
model.train(train_data, train_labels_encoded, epochs=100, lr=0.0015)

# Make predictions - returns actual wine ratings (3-9)
test_pred_ratings = model.forward(test_data)

print(f"\nPredictions shape: {test_pred_ratings.shape}")
print(f"Labels shape: {test_labels.shape}")

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_pred_ratings)
print(f"Accuracy: {accuracy:.4f}")

print("\nSample Predictions vs Actual:")
for i in range(10):
    print(f"Actual: {test_labels[i]}, Predicted: {test_pred_ratings[i]}")

# Show raw predictions for first few samples
print("\nRaw Predictions (first 5 samples):")
raw_preds = model.predict_proba(test_data[:5])
quality_labels = ['Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']

for i in range(5):
    print(f"Sample {i+1}:")
    for j, pred_val in enumerate(raw_preds[i]):
        print(f"  {quality_labels[j]}: {pred_val:.4f}")
    print(f"  Predicted: {test_pred_ratings[i]}, Actual: {test_labels[i]}")
    print()

# Show distribution of predictions
print("Prediction Distribution:")
unique, counts = np.unique(test_pred_ratings, return_counts=True)
for quality, count in zip(unique, counts):
    print(f"Quality {quality}: {count} predictions")
