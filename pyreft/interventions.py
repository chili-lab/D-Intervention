import torch
from collections import OrderedDict
import math

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from pyvene.models.layers import LowRankRotateLayer
from transformers.activations import ACT2FN


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return


class NoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LobiReFT(h) = h + R^T(b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))
    
class VIBreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    VIB-ReFT: Variational Information Bottleneck version of ReFT
    Adds stochastic sampling with learned mean and variance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        # add compression flag and beta 
        self.use_compression = kwargs.get('use_compression', False)
        self.use_norm = kwargs.get('use_norm', False)
        self.beta = kwargs.get('beta', 0.1)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        if self.use_norm == True:
            self.norm = RMSNorm(self.embed_dim.item()).to(
                kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        # Separate networks for mean and log variance
        self.mean_network = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.logvar_network = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
            
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) with constrained std"""
        # Clamp logvar to prevent numerical instability
        # New range allows std between ~0.001 and ~0.2
        logvar = torch.clamp(logvar, min=-14.0, max=-3.2)
        
        std = torch.exp(0.5 * logvar)
        # Clamp std directly to reasonable values based on weight distribution
        std = torch.clamp(std, min=0.001, max=0.2)
        
        # Scale random noise to be proportional to weight distribution
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def compute_kl_divergence(self, mu, logvar):
        """Compute KL divergence between N(mu, var) and N(0, 1)"""
        # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        if self.use_norm == True:
            print("Using RMSNorm")
            base = self.norm(base)
        rotated_base = self.rotate_layer(base)
        
        # Get mean and variance
        mu = self.act_fn(self.mean_network(base))
        logvar = self.logvar_network(base)
        
        # Sample using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        output = base + torch.matmul(
            (z - rotated_base), self.rotate_layer.weight.T
        )
        
        # Store mu and logvar for loss computation
        self.last_mu = mu
        self.last_logvar = logvar
        
        return self.dropout(output.to(base.dtype))
    
class VIBAffinereftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    VIB-ReFT: Variational Information Bottleneck version of ReFT
    Adds stochastic sampling with learned mean and variance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        # add compression flag and beta 
        self.use_compression = kwargs.get('use_compression', False)
        self.use_norm = kwargs.get('use_norm', True)
        self.beta = kwargs.get('beta', 0.1)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.d_type = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        self.norm_eps = 1e-6
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        if self.use_norm == True:
            self.norm = RMSNorm(self.embed_dim.item(), eps=self.norm_eps).to(
                self.d_type)

        # Replace simple linear layers with SwiGLU encoders for mean and logvar
        hidden_dim = kwargs.get("hidden_dim", kwargs["low_rank_dimension"] * 4)
        
        # SwiGLU components for mean network
        self.mean_w1 = torch.nn.Linear(self.embed_dim, hidden_dim).to(self.d_type)
        self.mean_w2 = torch.nn.Linear(self.embed_dim, hidden_dim).to(self.d_type)
        self.mean_w3 = torch.nn.Linear(hidden_dim, kwargs["low_rank_dimension"]).to(self.d_type)
        
        # SwiGLU components for logvar network
        self.logvar_w1 = torch.nn.Linear(self.embed_dim, hidden_dim).to(self.d_type)
        self.logvar_w2 = torch.nn.Linear(self.embed_dim, hidden_dim).to(self.d_type)
        self.logvar_w3 = torch.nn.Linear(hidden_dim, kwargs["low_rank_dimension"]).to(self.d_type)
        
        self.act_fn = ACT2FN["silu"]  # SwiGLU uses SiLU activation
        
    def encode_mean(self, x):
        # SwiGLU computation for mean
        gate = self.act_fn(self.mean_w1(x))
        value = self.mean_w2(x)
        hidden = gate * value
        return self.mean_w3(hidden)
        
    def encode_logvar(self, x):
        # SwiGLU computation for logvar
        gate = self.act_fn(self.logvar_w1(x))
        value = self.logvar_w2(x)
        hidden = gate * value
        return self.logvar_w3(hidden)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) with constrained std"""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        # Additional clamp on std for extra safety
        std = torch.clamp(std, min=1e-6, max=1.0)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, base, source=None, subspaces=None
    ):
        
        base = self.norm(base)
        
        # Use SwiGLU encoders
        mu = self.encode_mean(base)
        logvar = self.encode_logvar(base)
        
        rotated_base = self.rotate_layer(base)
        
        # Rest of the forward pass remains the same
        z = self.reparameterize(mu, logvar)
        output = base + torch.matmul(
            (z - rotated_base), self.rotate_layer.weight.T
        )
        
        self.last_mu = mu
        self.last_logvar = logvar
        
        return output.to(base.dtype)

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Overwrite for data-efficiency.
    #     """
    #     state_dict = OrderedDict()
    #     for k, v in self.mean_network.state_dict().items():
    #         state_dict[f"mean_{k}"] = v
    #     for k, v in self.logvar_network.state_dict().items():
    #         state_dict[f"logvar_{k}"] = v
    #     state_dict["rotate_layer"] = self.rotate_layer.weight.data
    #     return state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """
    #     Overwrite for data-efficiency.
    #     """
    #     mean_dict = OrderedDict()
    #     logvar_dict = OrderedDict()
        
    #     # Split state dict into mean and logvar components
    #     for k, v in state_dict.items():
    #         if k.startswith("mean_"):
    #             mean_dict[k[5:]] = v
    #         elif k.startswith("logvar_"):
    #             logvar_dict[k[7:]] = v
                
    #     self.mean_network.load_state_dict(mean_dict, strict=False)
    #     self.logvar_network.load_state_dict(logvar_dict, strict=False)
        
    #     if "rotate_layer" in state_dict:
    #         overload_w = state_dict["rotate_layer"]
    #         overload_w_width = overload_w.shape[-1]
    #         self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
    #     return

class VIBRawreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    VIB-RawReFT: Variational Information Bottleneck version of RawReFT
    Adds stochastic sampling with learned mean and variance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        # add compression flag and beta 
        self.use_compression = kwargs.get('use_compression', False)
        self.use_residual = kwargs.get('use_residual', False)
        self.beta = kwargs.get('beta', 0.1)
        self.decoder_layer = torch.nn.Linear(kwargs["low_rank_dimension"],
            self.embed_dim).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
            
        # Separate networks for mean and log variance
        self.mean_network = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.logvar_network = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
            
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) with constrained std"""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        # Additional clamp on std for extra safety
        std = torch.clamp(std, min=1e-6, max=1.0)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def compute_kl_divergence(self, mu, logvar):
        """Compute KL divergence between N(mu, var) and N(0, 1)"""
        # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()
        
    def forward(
        self, base, source=None, subspaces=None
    ):      
        # Get mean and variance
        mu = self.mean_network(base)
        logvar = self.logvar_network(base)
        
        # Sample using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        if self.use_residual == False:
            output = self.decoder_layer(z)
        else:
            output = base + self.decoder_layer(z)
        
        # Store mu and logvar for loss computation
        self.last_mu = mu
        self.last_logvar = logvar
        
        return self.dropout(output.to(base.dtype))
    
    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
       
class MiniTransformerIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    A minimal transformer-based intervention that applies self-attention
    and feedforward processing to the input embeddings.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        
        # # Convert embed_dim to int if it's a tensor
        # self.embed_dim = int(self.embed_dim) if torch.is_tensor(self.embed_dim) else self.embed_dim
        
        # Define dimensions
        self.hidden_dim = kwargs["low_rank_dimension"]
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.d_type = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        self.norm_eps = 1e-6
        
        # Multi-head attention components
        self.q_proj = torch.nn.Linear(self.embed_dim, self.hidden_dim).to(
            self.d_type)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.hidden_dim).to(
            self.d_type)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.hidden_dim).to(
            self.d_type)
        self.o_proj = torch.nn.Linear(self.hidden_dim, self.embed_dim).to(
            self.d_type)
        
        # Replace feedforward network with SWiGLU components
        self.w1 = torch.nn.Linear(self.embed_dim, self.hidden_dim * 4).to(self.d_type)
        self.w2 = torch.nn.Linear(self.embed_dim, self.hidden_dim * 4).to(self.d_type)
        self.w3 = torch.nn.Linear(self.hidden_dim * 4, self.embed_dim).to(self.d_type)
        
        # Layer normalization with integer dimension
        self.norm1 = RMSNorm(self.embed_dim.item(),eps=self.norm_eps).to(
            self.d_type)
        self.norm2 = RMSNorm(self.embed_dim.item(),eps=self.norm_eps).to(
            self.d_type)
        
        # Dropout
        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.1))
        self.act_fn = ACT2FN["silu"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def attention(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)
        
        return self.o_proj(out)
        
    def forward(self, base, source=None, subspaces=None):
        # Self-attention block
        residual = base
        x = self.norm1(base)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x
        
        # Modified feedforward block with SWiGLU
        residual = x
        x = self.norm2(x)
        
        # SWiGLU computation
        gate = self.act_fn(self.w1(x))  # SiLU activation
        value = self.w2(x)
        x = gate * value  # Element-wise multiplication
        
        x = self.w3(x)  # Project back to original dimension
        x = self.dropout(x)
        x = residual + x
        
        return x.to(base.dtype)

class RedIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    RED(h) = Rh + b
    Simple rotation with bias intervention
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_bias = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = torch.matmul(
            self.rotate_layer(base) + self.learned_bias, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

class VIBLobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Variational version of LobiReFT with reparameterization trick
    VIB-LobiReFT(h) = h + R^T(z), where z ~ N(μ, σ²)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        
        # Replace single parameter with mean and logvar parameters
        self.mean = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.logvar = torch.nn.Parameter(
            torch.zeros(kwargs["low_rank_dimension"]), requires_grad=True)
        
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) with constrained std"""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        # Additional clamp on std for extra safety
        std = torch.clamp(std, min=1e-6, max=1.0)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        # Sample z using reparameterization trick
        z = self.reparameterize(self.mean, self.logvar)
        
        output = base + torch.matmul(
            z, self.rotate_layer.weight.T
        )
        
        # Store for potential loss computation
        self.last_mu = self.mean
        self.last_logvar = self.logvar
        
        return self.dropout(output.to(base.dtype))

class VIBRedIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Variational version of RED with reparameterization trick
    VIB-RED(h) = Rh + z, where z ~ N(μ, σ²)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        
        # Replace single bias with mean and logvar parameters
        self.mean = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.logvar = torch.nn.Parameter(
            torch.zeros(kwargs["low_rank_dimension"]), requires_grad=True)
            
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) with constrained std"""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        # Additional clamp on std for extra safety
        std = torch.clamp(std, min=1e-6, max=1.0)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        # Sample z using reparameterization trick
        z = self.reparameterize(self.mean, self.logvar)
        
        output = torch.matmul(
            self.rotate_layer(base) + z, self.rotate_layer.weight.T
        )
        
        output = torch.matmul(
            self.rotate_layer(base) + self.learned_bias, self.rotate_layer.weight.T
        )
        
        # Store for potential loss computation
        self.last_mu = self.mean
        self.last_logvar = self.logvar
        
        return self.dropout(output.to(base.dtype))
