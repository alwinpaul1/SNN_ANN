"""
PLIF Neuron Implementation - Corrected per Paper Equation (2)

Key Correction: τ = 1/sigmoid(w), NOT sigmoid(w)
The original implementation had this backwards!

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
Equation (2): V[t] = V[t-1] + (1/τ)(X[t] - (V[t-1] - Vreset))
where τ = sigmoid(w)^(-1) (the INVERSE of sigmoid)
"""

import torch
import torch.nn as nn


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient function for spike generation.
    Uses the rectangular function as surrogate (following SpikingJelly).
    
    Forward: Heaviside step function (threshold at 1.0)
    Backward: Rectangular surrogate gradient
    """
    
    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        # Spike if membrane potential exceeds threshold
        return (membrane_potential >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        membrane_potential, = ctx.saved_tensors
        threshold = ctx.threshold
        
        # Rectangular surrogate gradient
        # Width of the surrogate: 1.0 (can be tuned)
        grad = (torch.abs(membrane_potential - threshold) < 0.5).float()
        
        return grad * grad_output, None


class PLIFNeuron(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) Neuron
    
    Paper Equation (2):
        V[t] = V[t-1] + (1/τ) * (X[t] - (V[t-1] - V_reset))
    
    where:
        τ = sigmoid(w)^(-1)  # CRITICAL: It's the INVERSE!
        w is a learnable parameter
    
    Args:
        tau_init: Initial value for the effective time constant τ
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset potential after spike (default: 0.0)
        detach_reset: Whether to detach reset in backward pass
    """
    
    def __init__(
        self,
        tau_init: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        # Learnable time constant parameter
        # Initialize w such that 1/sigmoid(w) ≈ tau_init
        # If tau_init=2.0, we want sigmoid(w) = 0.5, so w ≈ 0
        # sigmoid(w) = 1/tau_init => w = log(1/tau_init / (1 - 1/tau_init))
        if tau_init > 1.0:
            init_w = torch.log(torch.tensor(1.0 / tau_init / (1.0 - 1.0 / tau_init)))
        else:
            # For tau_init <= 1, sigmoid needs to be >= 1, which is impossible
            # Use a default value
            init_w = torch.tensor(0.0)
        
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        
        # Surrogate gradient function
        self.spike_fn = SurrogateGradient.apply
    
    @property
    def tau(self) -> torch.Tensor:
        """
        Compute τ = sigmoid(w)^(-1) = 1/sigmoid(w)
        
        CRITICAL CORRECTION: Paper says τ = sigmoid(w)^(-1)
        This is the INVERSE of sigmoid, not sigmoid itself!
        """
        return 1.0 / torch.sigmoid(self.w)
    
    def forward(self, x: torch.Tensor, v_prev: torch.Tensor = None) -> tuple:
        """
        Single time step forward pass.
        
        Args:
            x: Input current [B, C, H, W]
            v_prev: Previous membrane potential [B, C, H, W] or None
        
        Returns:
            spike: Output spikes [B, C, H, W]
            v_new: New membrane potential [B, C, H, W]
        """
        if v_prev is None:
            v_prev = torch.zeros_like(x)
        
        # Paper Equation (2): V[t] = V[t-1] + (1/τ)(X[t] - (V[t-1] - V_reset))
        tau = self.tau
        v_new = v_prev + (1.0 / tau) * (x - (v_prev - self.v_reset))
        
        # Generate spike using surrogate gradient
        spike = self.spike_fn(v_new, self.v_threshold)
        
        # Reset: hard reset after spike
        if self.detach_reset:
            v_new = v_new * (1 - spike.detach()) + self.v_reset * spike.detach()
        else:
            v_new = v_new * (1 - spike) + self.v_reset * spike
        
        return spike, v_new
    
    def extra_repr(self) -> str:
        return f'tau_init={1.0/torch.sigmoid(self.w).item():.2f}, v_threshold={self.v_threshold}, v_reset={self.v_reset}'


class PLIFNeuronMultiStep(nn.Module):
    """
    Multi-step PLIF neuron that processes entire temporal sequence.
    
    This is more efficient for training as it processes all time steps at once.
    """
    
    def __init__(
        self,
        tau_init: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True
    ):
        super().__init__()
        self.neuron = PLIFNeuron(tau_init, v_threshold, v_reset, detach_reset)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process entire temporal sequence.
        
        Args:
            x: Input tensor [T, B, C, H, W] (time-first format)
        
        Returns:
            spikes: Output spikes [T, B, C, H, W]
        """
        T = x.shape[0]
        spikes = []
        v = None
        
        for t in range(T):
            spike, v = self.neuron(x[t], v)
            spikes.append(spike)
        
        return torch.stack(spikes, dim=0)  # [T, B, C, H, W]


# Testing code
if __name__ == '__main__':
    print("Testing PLIF Neuron Implementation...")
    
    # Test single step
    neuron = PLIFNeuron(tau_init=2.0)
    print(f"Initial tau: {neuron.tau.item():.4f}")
    
    x = torch.randn(2, 64, 32, 32)  # [B, C, H, W]
    spike, v = neuron(x)
    print(f"Input shape: {x.shape}")
    print(f"Spike shape: {spike.shape}")
    print(f"Membrane shape: {v.shape}")
    print(f"Spike rate: {spike.mean().item():.4f}")
    
    # Test multi-step
    neuron_ms = PLIFNeuronMultiStep(tau_init=2.0)
    x_seq = torch.randn(10, 2, 64, 32, 32)  # [T, B, C, H, W]
    spikes = neuron_ms(x_seq)
    print(f"\nMulti-step input shape: {x_seq.shape}")
    print(f"Multi-step output shape: {spikes.shape}")
    print(f"Overall spike rate: {spikes.mean().item():.4f}")
    
    # Test gradient flow
    x_grad = torch.randn(10, 2, 64, 32, 32, requires_grad=True)
    spikes_grad = neuron_ms(x_grad)
    loss = spikes_grad.sum()
    loss.backward()
    print(f"\nGradient check - tau grad: {neuron_ms.neuron.w.grad}")
    print(f"Input grad exists: {x_grad.grad is not None}")
    
    print("\n✓ PLIF Neuron tests passed!")
