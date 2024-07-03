import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 100 # max epoch





class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

# def mem_update(ops, x, mem, spike):
#     mem = mem * decay * (1. - spike) + ops(x) 
#     spike = act_fun(mem) # act_fun : approximation firing function
#     return mem, spike

def mem_update(y, mem, spike, time_step):
    # odd and even indices multiplex
    # y = ops(x)

    # mask_odd = torch.zeros(y.shape,device=device)
    # mask_even = torch.zeros(y.shape,device=device)
    # mask_odd[:,1::2] = 1
    # mask_even[:,::2] = 1
    # mask_odd = mask_odd.bool()
    # mask_even = mask_even.bool()

    # y_even = torch.masked_select(y,mask_even).view(batch_size,-1)
    # y_odd = torch.masked_select(y,mask_odd).view(batch_size,-1)
    # is_odd = time_step % 2
    # y = ops(x)
    # print(mem.shape)
    # print(spike.shape)
    # print(y.shape)
    mem = mem * decay * (1. - spike) + y
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

    

    

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [128, 10]

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.input_layer = 784
        self.hidden_layer = 60
        self.output_layer = 10
        self.muliplex_hidden_layer = 30
        self.muliplex_output_layer = 5

        self.fc1 = nn.Linear(self.input_layer,self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer,self.output_layer)


    def forward(self, input, time_window = 20):

        h_spike = torch.zeros(batch_size,self.hidden_layer,device=device)
        o_spike = o_sum = torch.zeros(batch_size,self.output_layer,device=device)

        h_mem = torch.zeros(batch_size,self.muliplex_hidden_layer,device=device)
        o_mem = torch.zeros(batch_size,self.muliplex_output_layer,device=device)

        self.mask_hidden_odd = torch.zeros((60,30), device=device, requires_grad=False)
        self.mask_hidden_even = torch.zeros((60,30), device=device, requires_grad=False)
        self.mask_hidden_odd[1::2,:] = torch.eye(30).clone()
        self.mask_hidden_even[::2,:] = torch.eye(30).clone()

        self.mask_out_odd = torch.zeros((10,5), device=device, requires_grad=False)
        self.mask_out_even = torch.zeros((10,5), device=device, requires_grad=False)
        self.mask_out_odd[1::2,:] = torch.eye(5).clone()
        self.mask_out_even[::2,:] = torch.eye(5).clone()


        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing
            
            if step % 2 == 0:   # even indices
                yh = self.fc1(x.float())
                h_mem, h_spike_even = mem_update(yh @ self.mask_hidden_even, h_mem, h_spike @ self.mask_hidden_even, step)
                # h_sum += h_spike_even
                h_spike = h_spike_even @ self.mask_hidden_even.T

                yo = self.fc2(h_spike)
                o_mem, o_spike_even = mem_update(yo @ self.mask_out_even, o_mem, o_spike @ self.mask_out_even, step)
                o_sum = (o_sum @ self.mask_out_even + o_spike_even) @ self.mask_out_even.T
                o_spike = o_spike_even @ self.mask_out_even.T

            else:               # odd indices
                yh = self.fc1(x.float())
                h_mem, h_spike_odd = mem_update(yh @ self.mask_hidden_odd, h_mem, h_spike @ self.mask_hidden_odd, step)
                # h_sum += h_spike_odd
                h_spike = h_spike_odd @ self.mask_hidden_odd.T

                yo = self.fc2(h_spike)
                o_mem, o_spike_odd = mem_update(yo @ self.mask_out_odd, o_mem, o_spike @ self.mask_out_odd, step)
                o_sum = (o_sum @ self.mask_out_odd + o_spike_odd) @ self.mask_out_odd.T
                o_spike = o_spike_odd @ self.mask_out_odd.T

        outputs = o_sum / time_window
        return outputs