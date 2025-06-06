# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM

To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## THEORY

Recurrent Neural Networks (RNNs) are a specialized class of artificial neural networks designed to process sequential data. Unlike traditional feedforward neural networks, RNNs incorporate feedback connections, enabling them to maintain a memory of previous inputs across time steps. This makes them particularly well-suited for time series forecasting tasks, such as stock price prediction. By capturing temporal dependencies within historical data, RNNs can learn complex patterns and trends, allowing for more accurate modeling of future price movements. Their ability to leverage prior context makes them effective for dynamic and non-linear datasets commonly encountered in financial markets.

## DESIGN STEPS

### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: VIKASH A R

### Register Number: 212222040179

```python
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
    self.fc = nn.Linear(hidden_size,output_size)
  def forward(self, x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

# Train the Model

def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()  # Clear previous gradients
      outputs = model(x_batch)  # Forward pass
      loss = criterion(outputs, y_batch)  # Compute loss
      loss.backward()  # Backpropagation
      optimizer.step()  # Update weights
      total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    # Plot training loss
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model(model,train_loader,criterion,optimizer)
```

### OUTPUT

## Training Loss Over Epochs Plot

![image](https://github.com/user-attachments/assets/8dd4c378-a64f-43d8-a70e-415e66f87c23)

## True Stock Price, Predicted Stock Price vs time

![image](https://github.com/user-attachments/assets/1a1ec8b1-5036-4ca5-abf2-d141a563a2b3)

### Predictions
![image](https://github.com/user-attachments/assets/c6a5b03f-ede4-4ed8-b0e9-511e41bdb4b2)

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
