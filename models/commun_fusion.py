"""Implements common fusion patterns."""
"""link of the code source : https://github.com/pliang279/MultiBench"""
import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ArithmeticFusion(nn.Module):
    def __init__(self,hidden_dim,num_modalities,operation:str="add",add_concat=True):
        super(ArithmeticFusion,self).__init__()
        self.operation = getattr(torch, operation)
        self.add_concat = add_concat
        self.layers = nn.ModuleList([nn.LazyLinear(hidden_dim) for n in range(num_modalities)])


    def forward(self,modalities):
            outs =[]
            for i,modality in enumerate(modalities):
                out=torch.relu(self.layers[i](modality))
                outs.append(out)
            if len(outs)==2:
                z = self.operation(outs[0],outs[1])
            else:
                z = self.operation(outs[0],outs[1])
                for i in range(2,len(outs)):
                    z=self.operation(z,outs[i])
            # print("len outs ",len(outs))
            # print("shape 0",outs[0].shape)
            # print("z shape",z.shape)
            # print("add concat",self.add_concat)
            out =  z if not self.add_concat else torch.cat(outs + [z],dim=1)
           # print("out shape ",out.shape)
            return out



class ModalityAttention(nn.Module):
    def __init__(self):
        super(ModalityAttention, self).__init__()
        # learnable parameters (initialized later)
        self.w1 = None
        self.w2 = None
        self.normalizing_dim = None
        
    def init_params(self, modalities):
        # set hidden size based on input tensors
        self.hidden_dim1 = modalities[0].shape[-1]
        self.hidden_dim2 = modalities[1].shape[-1]
        
        # set normalizing dimension
        self.normalizing_dim = nn.Linear(self.hidden_dim2, self.hidden_dim1, bias=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # initialize learnable parameters
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_dim1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.w1.data.uniform_(-0.1, 0.1)

        self.w2 = nn.Parameter(torch.Tensor(self.hidden_dim1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.w2.data.uniform_(-0.1, 0.1)
        

    def forward(self, modalities):
        assert len(modalities) == 2, "ModalityAttention only supports 2 modalities"
        #assert modalities[0].shape[-1]==modalities[1].shape[-1], "ModalityAttention only supports modalities with same hidden dimension"  
        # initialize parameters if not already initialized
        if self.w1 is None and self.w2 is None and self.normalizing_dim is None:
            self.init_params(modalities)
        image_emb = modalities[0]
        text_emb = self.normalizing_dim(modalities[1])
       

    
        # compute attention scores
        e_image = torch.matmul(image_emb, self.w1)
        e_text = torch.matmul(text_emb, self.w2)
        
        # compute weight using softmax
        lambda_ = torch.exp(e_image) / (torch.exp(e_image) + torch.exp(e_text))
        lambda_ = lambda_.unsqueeze(1)
       
        # compute aggregated embedding
        c = lambda_ * image_emb + (1 - lambda_) * text_emb
        return c


class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
      #      print("modality shape ",modality.shape)
            flattened.append(torch.flatten(modality, start_dim=1))
        out = torch.cat(flattened, dim=1)
      #  print("out shape",out.shape)
        return out



class ConcatEarly(nn.Module):
    """Concatenation of input data on dimension 2."""

    def __init__(self):
        """Initialize ConcatEarly Module."""
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of ConcatEarly.
        
        :param modalities: An iterable of modalities to combine
        """
      
        return torch.cat(modalities, dim=2)

class CustomConcat(nn.Module):
    def __init__(self,hidden_dim,num_modalities):
        super(CustomConcat,self).__init__()
        
        #self.layers = [nn.LazyLinear(hidden_dim).to(DEVICE) for n in range(num_modalities)]
        self.layers = nn.ModuleList([nn.LazyLinear(hidden_dim) for n in range(num_modalities)])
        #self.layers = nn.ModuleList([nn.Linear(512,hidden_dim),nn.Linear(798,hidden_dim)])
    
    def forward(self,modalities):
            outs =[]
            for i,modality in enumerate(modalities):
                out=torch.relu(self.layers[i](modality))
                outs.append(out)
            out=torch.cat(outs,1)
      #      print("custom concat out shape",out.shape)
            return torch.cat(outs,1)
            
# Stacking modalities
class Stack(nn.Module):
    """Stacks modalities together on dimension 1."""

    def __init__(self):
        """Initialize Stack Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.stack(flattened, dim=2)

class ConcatWithLinear(nn.Module):
    """Concatenates input and applies a linear layer."""

    def __init__(self,hidden_dim, concat_dim=1):
        """Initialize ConcatWithLinear Module.
        
        :param input_dim: The input dimension for the linear layer
        :param hidden_dim: The output dimension for the linear layer
        :concat_dim: The concatentation dimension for the modalities.
        """
        super(ConcatWithLinear, self).__init__()
        self.concat_dim = concat_dim
        self.fc = nn.LazyLinear(hidden_dim)

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        return self.fc(torch.cat(modalities, dim=self.concat_dim))


class MultiplicativeInteractions3Modal(nn.Module):
    """Implements 3-Way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, task=None):
        """Initialize MultiplicativeInteractions3Modal object.

        :param input_dims: list or tuple of 3 integers indicating sizes of input
        :param output_dim: size of outputs
        :param task: Set to "affect" when working with social data.
        """
        super(MultiplicativeInteractions3Modal, self).__init__()
        self.a = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  [input_dims[2], output_dim], 'matrix3D')
        self.b = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  output_dim, 'matrix')
        self.task = task

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions3Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if self.task == 'affect':
            return torch.einsum('bm, bmp -> bp', modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])
        return torch.matmul(modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])


class MultiplicativeInteractions2Modal(nn.Module):
    """Implements 2-way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        """
        :param input_dims: list or tuple of 2 integers indicating input dimensions of the 2 modalities
        :param output_dim: output dimension
        :param output: type of MI, options from 'matrix3D','matrix','vector','scalar'
        :param flatten: whether we need to flatten the input modalities
        :param clip: clip parameter values, None if no clip
        :param grad_clip: clip grad values, None if no clip
        :param flip: whether to swap the two input modalities in forward function or not
        
        """
        super(MultiplicativeInteractions2Modal, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == 'matrix3D':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                input_dims[0], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(
                input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.b)

        # most general Hypernetworks as Multiplicative Interactions.
        elif output == 'matrix':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        # Diagonal Forms and Gating Mechanisms.
        elif output == 'vector':
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        # Scales and Biases.
        elif output == 'scalar':
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(
                    grad, grad_clip[0], grad_clip[1]))

    def _repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions2Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == 'matrix3D':
            Wprime = torch.einsum('bn, nmpq -> bmpq', m1,
                                  self.W) + self.V  # bmpq
            bprime = torch.einsum('bn, npq -> bpq', m1,
                                  self.U) + self.b    # bpq
            output = torch.einsum('bm, bmpq -> bpq', m2,
                                  Wprime) + bprime   # bpq

        # Hypernetworks as Multiplicative Interactions.
        elif self.output == 'matrix':
            Wprime = torch.einsum('bn, nmd -> bmd', m1,
                                  self.W) + self.V      # bmd
            bprime = torch.matmul(m1, self.U) + self.b      # bmd
            output = torch.einsum('bm, bmd -> bd', m2,
                                  Wprime) + bprime             # bmd

        # Diagonal Forms and Gating Mechanisms.
        elif self.output == 'vector':
            Wprime = torch.matmul(m1, self.W) + self.V      # bm
            bprime = torch.matmul(m1, self.U) + self.b      # b
            output = Wprime*m2 + bprime             # bm

        # Scales and Biases.
        elif self.output == 'scalar':
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self._repeatHorizontally(
                Wprime, self.input_dims[1]) * m2 + self._repeatHorizontally(bprime, self.input_dims[1])
        return output


class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    
    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(self, output_dim, rank, flatten=True):
        """
        Initialize LowRankTensorFusion object.
        
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten
        self.factors = []

        
        
    def init_factor(self,modalities):
        input_dims = [modality.shape[-1] for modality in modalities]
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim+1, self.output_dim)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1, self.output_dim)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # init the fusion weights
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        # low-rank factors
        if len(self.factors)==0:
            self.init_factor(modalities)
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
      #  print("lowrank output shape ",output.shape)
        return output

