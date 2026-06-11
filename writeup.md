# Sumcheck-Based ZKML Proving for ResNet-18 on CIFAR-10

## 1. System Architecture

### 1.1 Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  fusion_export.py│────▶│  fused_weights/  │────▶│   C++ Backend    │
│  (offline)       │     │  *.bin + meta     │     │   (zkresnet)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                    │
                                           ┌────────┴────────┐
                                           │                 │
                                      ┌────▼────┐     ┌─────▼─────┐
                                      │ Prover  │     │ Verifier  │
                                      └─────────┘     └───────────┘
```

Two phases:

1. **Offline Preprocessing** (`fusion_export.py`): Load a CIFAR-10 ResNet-18, algebraically fuse all BatchNorm layers into their preceding Conv2d layers, verify numerical equivalence, serialize the fused weights as flat little-endian `float32` binaries.

2. **Online Proving** (`zkresnet`): Ingest fused weight binaries and a quantized CIFAR-10 input, execute forward pass over $\mathbb{F}_p$ ($p = 2^{61} - 1$), construct the layered arithmetic circuit, run GKR-style sumcheck layer-by-layer with claim folding at residual addition nodes.

### 1.2 ResNet-18 Representation

| Stage | Layers | Output Shape |
|-------|--------|-------------|
| stem | Conv2d(3→64, 3×3), ReLU | 64×32×32 |
| layer1 | 2× BasicBlock(64→64) | 64×32×32 |
| layer2 | 2× BasicBlock(64→128, stride=2) | 128×16×16 |
| layer3 | 2× BasicBlock(128→256, stride=2) | 256×8×8 |
| layer4 | 2× BasicBlock(256→512, stride=2) | 512×4×4 |
| head | AdaptiveAvgPool → FC(512→10) | 10 |

Each BasicBlock:

$$H(x) = \text{ReLU}(F(x) + S(x))$$

$F(x) = W_2 \ast \text{ReLU}(W_1 \ast x)$, $S(x) = x$ or $S(x) = W_s \ast x$.

Circuit gate types:

| Gate Type | Semantics |
|-----------|-----------|
| `CONV` | $y_j = \sum_{i,k} W_{j,i,k} \cdot x_{i+k} + b_j$ |
| `RELU` | $y_i = x_i \cdot \mathbb{1}[x_i > 0]$ |
| `ADD` | $y_i = x_i^{(a)} + x_i^{(b)}$ (triggers claim folding) |
| `AVGPOOL` | $y_i = k^{-2} \sum_{j \in \text{pool}(i)} x_j$ |
| `FC` | $y_j = \sum_i W_{j,i} \cdot x_i + b_j$ |
| `FLATTEN` | Identity permutation |

---

## 2. Offline BatchNorm Fusion

### 2.1 The Fusion Identity

BatchNorm with frozen statistics:

$$\text{BN}(z) = \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $z = W \ast x + b_{\text{conv}}$. Define $\lambda = \gamma / \sqrt{\sigma^2 + \epsilon}$. Substituting:

$$\text{BN}(W \ast x + b_{\text{conv}}) = \lambda W \ast x + \lambda(b_{\text{conv}} - \mu) + \beta$$

Fused parameters:

$$\boxed{W_{\text{fused}} = \lambda \cdot W, \quad b_{\text{fused}} = \lambda \cdot (b_{\text{conv}} - \mu) + \beta}$$

where $\lambda_j$ is applied per-output-channel: $W_{\text{fused}}[j, i, k_h, k_w] = \lambda_j \cdot W[j, i, k_h, k_w]$.

### 2.2 Soundness

BatchNorm during inference is an affine map $z \mapsto \lambda z + (\beta - \lambda\mu)$ per channel. Convolution is affine in $x$. Composition of affine maps is affine, represented exactly by the fused parameters. Numerical error is $O(\epsilon_{\text{mach}}) \approx 10^{-7}$ for `float32`. Verified empirically: $\|y_{\text{orig}} - y_{\text{fused}}\|_\infty < 10^{-4}$.

### 2.3 Why Not Arithmetize BatchNorm?

Arithmetizing $(\sigma^2 + \epsilon)^{-1/2}$ requires inverse square root in $\mathbb{F}_p$ — high-degree, needing additional witness columns and range proofs. This inflates circuit size by $\Theta(C_{\text{out}})$ per BN layer (20 layers). Offline fusion eliminates this at zero runtime cost.

---

## 3. Sumcheck Protocol

### 3.1 GKR Framework

For layered circuit with $L$ layers, verifier holds claim $\tilde{V}_L(r_L) = c_L$ and reduces layer-by-layer. At layer $\ell$:

$$\tilde{V}_\ell(g) = \sum_{u, v} \left[\widetilde{\text{add}}_\ell(g, u, v) \cdot (\tilde{V}_{\ell-1}(u) + \tilde{V}_{\ell-1}(v)) + \widetilde{\text{mult}}_\ell(g, u, v) \cdot \tilde{V}_{\ell-1}(u) \cdot \tilde{V}_{\ell-1}(v)\right]$$

### 3.2 Per-Layer Execution

For output bit-length $n$, input bit-length $m$, each sumcheck runs $m$ rounds. Round $j$:

1. Prover sends degree-2 polynomial $p_j(X_j)$.
2. Verifier checks $p_j(0) + p_j(1) = s_{j-1}$.
3. Verifier samples $r_j \leftarrow \mathbb{F}_p$, sets $s_j = p_j(r_j)$.

### 3.3 Bookkeeping Optimization

Prover maintains arrays `bk[i]`, `mt[i]`. Each round:

$$s_0 = \sum_i \texttt{bk}[2i] \cdot \texttt{mt}[2i]$$
$$s_1 = \sum_i \texttt{bk}[2i+1] \cdot \texttt{mt}[2i+1]$$
$$s_2 = \sum_i (2\texttt{bk}[2i+1] - \texttt{bk}[2i]) \cdot (2\texttt{mt}[2i+1] - \texttt{mt}[2i])$$

Monomial coefficients from $(s_0, s_1, s_2)$:

$$a = s_2 + s_0 - 2s_1, \quad b = s_1 - s_0, \quad c = s_0$$

After challenge $r_j$: $\texttt{bk}[i] \leftarrow \texttt{bk}[2i](1 - r_j) + \texttt{bk}[2i+1] r_j$

**Prover:** $O(m \cdot 2^m)$ per layer. **Verifier:** $O(m)$ per layer. **Proof:** $3m$ field elements per layer.

---

## 4. Residual Connection Claim Folding

### 4.1 The Exponential Blowup Problem

At ADD node $H(x) = F(x) + x$, verifier must verify two claims:
- $v_1 = \tilde{V}_F(r)$
- $v_2 = \tilde{V}_x(r)$

Naive forking: $2^d$ claims after $d$ blocks. ResNet-18 has 8 blocks → $2^8 = 256$ concurrent claims.

### 4.2 Folding

At each ADD node:

$$\boxed{v_{\text{folded}} = v_1 + \alpha \cdot v_2, \quad \alpha \leftarrow_R \mathbb{F}_p}$$

**Soundness (Schwartz–Zippel):** If $v_1 \neq \tilde{V}_F(r)$ or $v_2 \neq \tilde{V}_x(r)$:

$$\Pr_\alpha[v_1 + \alpha v_2 = \tilde{V}_F(r) + \alpha \tilde{V}_x(r)] \leq \frac{1}{|\mathbb{F}_p|} = \frac{1}{2^{61} - 1} \approx 2^{-61}$$

### 4.3 Claim Growth: $O(1)$

Each ADD node receives one incoming claim, produces one folded claim. The skip connection claim is absorbed, never propagated independently. For 8 ADD nodes: claim count remains 1 at every step.

### 4.4 Protocol at ADD Node

```
Verifier                          Prover
───────                          ──────
holds claim c = ṼL(r)
                                  computes v₁ = ṼF(r), v₂ = Ṽx(r)
                                  sends (v₁, v₂)
checks v₁ + v₂ = c
samples α ←R Fp, sends α
                                  v_folded = v₁ + α·v₂
```

Cost: 3 field elements per ADD node.

---

## 5. Field Arithmetic and Quantization

### 5.1 Mersenne-61

$p = 2^{61} - 1$. Multiplication: compute 128-bit product, then $a \cdot b \bmod p = \text{lo} + \text{hi}$ where $\text{lo} = r \& ((1 \ll 61) - 1)$, $\text{hi} = r \gg 61$.

**Limitation:** No pairing support. Polynomial commitment (Hyrax/KZG) requires BLS12-381. This prototype verifies sumcheck soundness only.

### 5.2 Fixed-Point Encoding

$$\text{encode}(x) = \lfloor x \cdot 2^{16} \rceil \bmod p$$

Negatives: $\text{encode}(-x) = p - \text{encode}(x)$. After bilinear ops, rescale by $(2^{16})^{-1} \bmod p$.

---

## 6. Profiling

| Metric | Measurement |
|--------|-------------|
| Prover Time | `chrono::high_resolution_clock` around `Prover::prove()` |
| Verifier Time | `chrono::high_resolution_clock` around `Verifier::verify()` |
| Proof Size | $3 \times 8$ bytes/round + $8$ bytes/final claim + $24$ bytes/fold |

Output: `RESULT,<prover_ms>,<verifier_ms>,<proof_bytes>,<ACCEPT|REJECT>`

---

## 7. Assumptions and Limitations

1. **Frozen BN only.** Training-time BN cannot be fused.
2. **ReLU sign heuristic.** Uses $x < p/2 \Rightarrow x \geq 0$ rather than auxiliary-bit decomposition.
3. **No zero-knowledge.** Proof is verifiable but not hiding.
4. **Mersenne-61.** $2^{61}$ soundness, not $2^{128}$. Production requires BLS12-381.
5. **No polynomial commitment.** Input-layer check omitted.
6. **Single image.** Batch size > 1 not supported.
7. **Fixed 3×32×32 input.** No dynamic shapes.

---

## 8. References

1. Liu, Xie, Zhang. *zkCNN: Zero Knowledge Proofs for CNN Predictions and Accuracy.* CCS 2021.
2. Goldwasser, Kalai, Rothblum. *Delegating Computation: Interactive Proofs for Muggles.* STOC 2008.
3. Thaler. *Time-Optimal Interactive Proofs for Circuit Evaluation.* CRYPTO 2013.
4. Wahby, Tzialla, Shelat, Thaler, Walfish. *Doubly-Efficient zkSNARKs Without Trusted Setup.* S&P 2018.
5. He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
