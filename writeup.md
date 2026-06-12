# writeup — sumcheck zkml resnet18

## graph representation

resnet18 gets unrolled into a flat DAG of layer nodes. each BasicBlock becomes CONV -> RELU -> CONV -> ADD -> RELU. the ADD nodes hold two input pointers (residual branch + skip) so the verifier knows where to trigger claim folding during the backward pass.

the python script handles all the pytorch-side stuff — loads pretrained resnet18, swaps the 7x7 stem to 3x3 (cifar10 is 32x32, the default stem would crush spatial dims), kills maxpool with nn.Identity(), fuses batchnorm, and dumps flat f32 binaries. the c++ backend just ingests those.

## batchnorm fusion

bn during inference is just an affine map per channel. let λ = γ / sqrt(σ² + ε). then:

- W_fused = λ · W
- b_fused = λ · (b_conv - μ) + β

this is exact, not an approximation. two affine ops composed = one affine op. we verify equivalence in the python script (atol < 1e-4 on a test input, actual error was ~3e-6).

the alternative — arithmetizing bn directly — would require computing (σ² + ε)^{-1/2} over F_p, which means high-degree constraints and range proofs for every bn layer. 20 bn layers in resnet18. not worth it when you can just fold it offline for free.

## residual claim folding

the core problem: at an ADD node H(x) = F(x) + x, the verifier gets two claims v1 and v2. if you fork the protocol naively you get 2^d claims after d residual blocks (2^8 = 256 for resnet18).

fix: fold immediately with a random challenge α sampled by the verifier.

v_folded = v1 + α · v2

by schwartz-zippel, cheating probability is 1/|F_p| ≈ 2^{-61}. claim count stays O(1) throughout the entire backward pass.

## profiling

ran on host cpu (rtx 4050 laptop), mersenne-61 field (p = 2^61 - 1):

- prover: 185.5 ms
- verifier: 33.2 ms  
- proof size: 7904 bytes
- graph: 49 layers, 22 sumcheck transcripts, 8 residual folds

## limitations

1. **relu bypass** — relu arithmetization needs bit decomposition / range proofs. we evaluate relu in the clear during witness gen and pass the eval point straight through in the sumcheck. a real implementation would swap to x² activation or do the aux bit stuff.

2. **hollowed conv arithmetization** — the sumcheck proves layer-to-layer MLE routing but doesn't bind the weight tensors into the proof. verifier never commits to g.weights. this is the biggest gap — a real conv arithmetization needs V_out = V_in · W.

3. **custom backend** — this is not a fork of TAMUCrypto/zkCNN. it's a from-scratch c++17 prover/verifier built to natively ingest the fused binaries. the tradeoff was speed of implementation vs using the full zkCNN framework with its heavier dependencies.

4. **no polynomial commitment** — stops at the sumcheck boundary. full snark would need kzg/hyrax for the input layer binding.

5. **cifar10 arch mods** — stem swapped to 3x3 conv, maxpool replaced with identity. this is standard practice for cifar10 resnet variants to preserve spatial resolution.

6. **field** — mersenne-61 is fast (single-word arithmetic, no gmp) but too small for real crypto security. production would use bls12-381.
