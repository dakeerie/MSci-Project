import torch

x = torch.randn(3, requires_grad=True) #must specify this extra argument for backward propagation
print(x)

y = x+2 #operation creates a computation graph for back propagation later
#Forward pass calculates output y
#Since we specify gradient, pytorch automatically creates function for us. Function used in back 
#propagation to calculate gradient. y has attribute grad_fn which points to gradient function dy/dx
print(y)
#Prints "AddBackward" because operation was addition above

z = y*y*2
print(z)
# z = z.mean()
# print(z)

#Different operations taken into account

v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
z.backward(v) #gives dz/dx
#no specification of vector jacobian product throws error as grad can only be created for scalar outputs
#vector jacobian product is just the same as a change of basis matrix transformation in terms of 
#derivatives of the original coordinates wrt new coordinates being transformed to.
print(x.grad)

#prevent pytorch from tracking history and calculating grad_fn attribute- if we want to adjust 
#network eg change weights. Prevent via:

t = torch.randn(3, requires_grad=True)
print(t)

# t.requires_grad(False)
# t.detach() returns new tensor (with same values) detached from the graph and doesn't require gradient
# with torch.no_grad(): then do our operations

t.requires_grad_(False) #trailing underscore modifies in place
# print(t)
t.requires_grad_(True)

o = t.detach()
print(t)
print(o)

with torch.no_grad():
    y = t + 2
    print(y)

#whenever we call the backwards function then the gradient for the tensor will be included and 
#summed up need to be careful if you don't want to include it in the network

#Example

weights = torch.ones(4, requires_grad=True)

for epoch in range(4):
    model_output = (weights*3).sum()

    model_output.backward() #vector jacobian product. Calculates gradient
    #Each iteration (given by range(i) assimilates gradient each time in the .backward() call 

    print(weights.grad)
    #before next iteration need to empty gradients
    weights.grad.zero_()
    #prints correct gradients again 
