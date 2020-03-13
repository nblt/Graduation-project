import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

mnist_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    
    
def fgsm(model, X, y, epsilon):
    # Construct FGSM adversarial examples on the examples X
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X+delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def plot_images(X, y, yp, M, N):
    # Draw M rows and N columns
    f,ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N,M*1.3))
    for i in range(M):
        for j in range(N):
            idx, predict = i*N+j, yp[i*N+j].max(dim=0)[1]
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap='gray')
            title = ax[i][j].set_title("Pred: {}".format(predict))
            plt.setp(title, color=('g' if predict==y[idx] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    
model_cnn_noise = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

def noise_epoch(loader, model, opt=None, lr=0.1, alpha=0.001, num_iters=10):
    """ Train a model whose parameters are robust to noise """
    
    total_loss, total_err = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
        if opt:
            # Save the current model parameters
            torch.save(model.state_dict(), "model_cnn_temporary.pt")
            
            # Use PGD to maximize the loss while the parameters change 
            # within a loo norm ball 
            for t in range(num_iters):
                for name, param in model.named_parameters():
                    if name == '0.weight' or name == '0.bais' or name == '2.weight' or name == '2.bais':
                        param.data += alpha*param.grad.detach().sign()
                    param.grad.zero_()
                    
                yp = model(X)
                loss = nn.CrossEntropyLoss()(yp, y)
                loss.backward()
            
            # Record the grad of w*
            grads = {}
            for name, param in model.named_parameters():
                #print (name, param.grad.detach())
                grads[name] = param.grad.detach().cpu()
                param.grad.zero_()
            
            # Recover the parameters and apply gradient descent using grad of w*
            model.load_state_dict(torch.load("model_cnn_temporary.pt", map_location='cuda:0'))
            
            for name, param in model.named_parameters():
                param.data -= grads[name].to(device) * lr
                param.grad.zero_()
                
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)
 

training = 1
model_name = 'model_cnn_noise_0.001_10_layer2.pt'

# Training
if training:
    print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
    for _ in range(10):
        lr = 0.1
        if _ > 3: lr = 0.01
        train_err, train_loss = noise_epoch(train_loader, model_cnn_noise, True, lr, 0.001, 10)
        test_err, test_loss = noise_epoch(test_loader, model_cnn_noise)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")    
    
    torch.save(model_cnn_noise.state_dict(), model_name)    


for X,y in test_loader:
    X,y = X.to(device), y.to(device)
    break
    
# Test
model_cnn_noise.load_state_dict(torch.load(model_name, map_location='cuda:0'))

# Add noise
for name, param in model_cnn_noise.named_parameters():
    if name == '0.weight':
        param.data += (torch.rand(param.data.shape) - 0.5) .cuda() * 0.
        
delta = pgd_linf(model_cnn_noise, X, y, epsilon=0.2, alpha=1e-2, num_iter=40)
yp = model_cnn_noise(X + delta)
# plot_images(X+delta, y, yp, 3, 6)

print ('Accuracy: ', np.array((yp.max(dim=1)[1] == y).cpu()).sum() / len(y))
