import numpy as np
from typing import Callable, Optional

# /C:/Users/akhil/submission.py

def mxfp4_moe_optimized(
    inputs: np.ndarray,
    gate_w: np.ndarray,
    gate_b: np.ndarray,
    expert_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
) -> np.ndarray:
    """
    Optimized MXFP4-style Mixture-of-Experts (Top-1) implementation using NumPy.

    Keeps the canonical MoE flow:
      input -> gating -> routing -> expert processing -> gather

    No CUDA, no atomics, no unsafe memory writes.
    Vectorized grouping achieved by sorting tokens by chosen expert to reduce boolean masks.

    Args:
      inputs: (M, D) float32 array of input tokens.
      gate_w: (D, E) float32 gating weights.
      gate_b: (E,) float32 gating biases.
      expert_fn: optional callable (tokens, expert_id) -> processed_tokens.
                 If None, the identity function is used.

    Returns:
      output: (M, D) array with expert outputs gathered back to original order.
    """
    # Basic checks and normalized dtypes
    inputs = np.asarray(inputs)
    gate_w = np.asarray(gate_w)
    gate_b = np.asarray(gate_b)

    assert inputs.ndim == 2, "inputs must be (M, D)"
    assert gate_w.ndim == 2, "gate_w must be (D, E)"
    assert gate_b.ndim == 1, "gate_b must be (E,)"

    M, D = inputs.shape
    Dw, E = gate_w.shape
    assert Dw == D, "gate_w first dim must match input feature dim"
    assert gate_b.shape[0] == E, "gate_b length must match number of experts"

    # Ensure float32 for predictable numeric behavior
    if inputs.dtype != np.float32:
        inputs = inputs.astype(np.float32)
    if gate_w.dtype != np.float32:
        gate_w = gate_w.astype(np.float32)
    if gate_b.dtype != np.float32:
        gate_b = gate_b.astype(np.float32)

    # Default expert: identity passthrough
    if expert_fn is None:
        def _identity(tokens: np.ndarray, expert_id: int) -> np.ndarray:
            return tokens.copy()
        expert_fn = _identity

    # Step 1: compute gating logits (M, E)
    logits = inputs @ gate_w + gate_b  # (M, E)

    # Step 2: top-1 routing
    routes = np.argmax(logits, axis=1).astype(np.int32)  # (M,)

    # If no tokens or single expert trivial case
    if M == 0:
        return np.zeros((0, D), dtype=np.float32)
    if E == 1:
        # All tokens go to expert 0
        out = expert_fn(inputs, 0)
        return out

    # Step 3: group tokens by expert using sorting (avoids many boolean masks)
    perm = np.argsort(routes)                 # indices that group by expert id
    routes_sorted = routes[perm]
    inputs_sorted = inputs[perm]

    # Count tokens per expert
    counts = np.bincount(routes_sorted, minlength=E)

    # Prepare output buffer for sorted tokens
    output_sorted = np.empty_like(inputs_sorted)

    # Step 4: process each expert on the contiguous slices
    ptr = 0
    for e in range(E):
        cnt = int(counts[e])
        if cnt == 0:
            continue
        start = ptr
        end = start + cnt
        tokens = inputs_sorted[start:end]

        # Expert processing (callable). Must return shape (cnt, D).
        expert_out = expert_fn(tokens, e)
        if expert_out.shape != tokens.shape:
            raise ValueError(f"expert_fn returned shape {expert_out.shape} for expert {e}, expected {tokens.shape}")

        output_sorted[start:end] = expert_out
        ptr = end

    # Step 5: scatter back to original order
    output = np.empty_like(output_sorted)
    output[perm] = output_sorted

    return output


# Example usage
if __name__ == "__main__":
    M, D, E = 128, 64, 4

    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((M, D)).astype(np.float32)
    gate_w = rng.standard_normal((D, E)).astype(np.float32)
    gate_b = np.zeros((E,), dtype=np.float32)

    out = mxfp4_moe_optimized(inputs, gate_w, gate_b)

    print("MoE output shape:", out.shape)

    

import numpy as np
from typing import Callable, Optional


def mxfp4_moe_optimized(
    inputs: np.ndarray,
    gate_w: np.ndarray,
    gate_b: np.ndarray,
    expert_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
) -> np.ndarray:

    inputs = np.asarray(inputs, dtype=np.float32)
    gate_w = np.asarray(gate_w, dtype=np.float32)
    gate_b = np.asarray(gate_b, dtype=np.float32)

    M, D = inputs.shape
    Dw, E = gate_w.shape

    assert Dw == D
    assert gate_b.shape[0] == E

    if expert_fn is None:
        def _identity(tokens: np.ndarray, expert_id: int) -> np.ndarray:
            return tokens.copy()
        expert_fn = _identity

    # Gating
    logits = inputs @ gate_w + gate_b
    routes = np.argmax(logits, axis=1).astype(np.int32)

    if M == 0:
        return np.zeros((0, D), dtype=np.float32)

    if E == 1:
        return expert_fn(inputs, 0)

    # Group by expert
    perm = np.argsort(routes)
    routes_sorted = routes[perm]
    inputs_sorted = inputs[perm]

    counts = np.bincount(routes_sorted, minlength=E)
    output_sorted = np.empty_like(inputs_sorted)

    ptr = 0
    for e in range(E):
        cnt = int(counts[e])
        if cnt == 0:
            continue

        start = ptr
        end = start + cnt

        tokens = inputs_sorted[start:end]
        expert_out = expert_fn(tokens, e)

        if expert_out.shape != tokens.shape:
            raise ValueError("Expert output shape mismatch")

        output_sorted[start:end] = expert_out
        ptr = end

    # Restore original order
    output = np.empty_like(output_sorted)
    output[perm] = output_sorted

    return output

def custom_kernel(inputs, gate_w, gate_b):
    return mxfp4_moe_optimized(inputs, gate_w, gate_b)


# Optional local test
if __name__ == "__main__":
    M, D, E = 128, 64, 4

    inputs = np.random.randn(M, D).astype(np.float32)
    gate_w = np.random.randn(D, E).astype(np.float32)
    gate_b = np.zeros((E,), dtype=np.float32)

    out = custom_kernel(inputs, gate_w, gate_b)

    print("MoE output shape:", out.shape)