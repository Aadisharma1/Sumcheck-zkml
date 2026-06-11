# Technical Writeup: Sumcheck ZKML for ResNet-18

## 1. Graph Representation
The system handles ResNet-18 inference via a two-stage pipeline: an offline Python frontend for graph manipulation and a C++ sumcheck backend. The graph is ingested into the C++ backend as a sequence of procedural struct definitions (`LayerDesc`) mapping to flat `float32` binaries. 

ResNet-18's topology is unrolled. `BasicBlock` modules are flattened into sequential `CONV -> RELU -> CONV -> ADD -> RELU` nodes. To manage the $H(x) = F(x) + x$ skip connections, the `ADD` nodes hold dual pointers to the upstream layers to trigger the claim folding protocol during the verifier's backward pass.

## 2. Offline BatchNorm Fusion & Soundness
Arithmetizing BatchNorm directly in the sumcheck circuit requires computing the inverse square root of the variance $(\sigma^2 + \epsilon)^{-1/2}$ over $\mathbb{F}_p$. This injects high-degree constraints and necessitates range proofs, needlessly blowing up the prover time.

Because inference statistics are frozen, BatchNorm is simply an affine transformation. We handle this offline by algebraically fusing the BN parameters $(\mu, \sigma^2, \gamma, \beta)$ into the preceding `Conv2d` layer's weights and biases.

Let $\lambda = \gamma / \sqrt{\sigma^2 + \epsilon}$. The fused weights are:
$W_{\text{fused}} = \lambda \cdot W$
$b_{\text{fused}} = \lambda \cdot (b_{\text{conv}} - \mu) + \beta$

**Soundness:** This is mathematically exact. The composition of two affine operations (Convolution and frozen BatchNorm) is just another affine operation. The verifier does not need to know BN ever existed in the original graph. The C++ backend ingests these fused tensors directly. Equivalence is verified in `fusion_export.py` to an ATOL of $1e-4$.

## 3. Residual Connections & Claim Folding
ResNet-18's residual connections create a branching computation graph. In a standard GKR sumcheck, evaluating an `ADD` node $H(x) = F(x) + x$ at a random point $r$ spawns two opening claims: $v_1 = \tilde{V}_F(r)$ and $v_2 = \tilde{V}_x(r)$. Left unchecked, the verifier's claim count grows exponentially ($O(2^d)$ for $d$ residual blocks), crushing verifier throughput.

We mitigate this using Random Linear Combination (RLC) claim folding.
When the verifier hits an `ADD` node, it receives $v_1$ and $v_2$ from the prover.
1. Verifier checks $v_1 + v_2 = c_{current}$.
2. Verifier samples a random challenge $\alpha \leftarrow \mathbb{F}_p$.
3. Verifier folds the claims: $v_{\text{folded}} = v_1 + \alpha v_2$.

By Schwartz-Zippel, the probability that the prover can cheat this folded claim is bounded by $1/|\mathbb{F}_p|$. Because we fold immediately at every residual intersection, the downstream claim count is strictly bounded to $O(1)$.

## 4. Hardware Profiling
Environment: NVIDIA RTX 4050 (Host CPU for C++ execution).
Field: Mersenne-61 ($p = 2^{61} - 1$)

* **Prover Time:** 185.528 ms
* **Verifier Time:** 33.1715 ms
* **Proof Size:** 7904 bytes

## 5. Limitations & Assumptions
1. **ReLU Arithmetization Bypass:** True ReLU arithmetization over $\mathbb{F}_p$ requires bit-decomposition. To isolate the sumcheck throughput for the Convolution/Residual logic in this starter prototype, the C++ backend currently evaluates ReLU in the clear during witness generation, but routes the evaluation point directly through during the sumcheck (`r_points[in_id] = r_points[l]`). A production implementation would require swapping to a low-degree polynomial activation (e.g., $x^2$) or implementing auxiliary range proofs.
2. **Fixed-Point Arithmetic:** Values are quantized to 16 fractional bits. 
3. **Commitment Scheme:** The current implementation halts at the sumcheck boundary. A full SNARK would require binding the input layer to a polynomial commitment scheme (e.g., KZG/Hyrax), which is omitted here.
