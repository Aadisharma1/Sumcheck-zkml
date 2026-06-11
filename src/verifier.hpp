#pragma once
#include "field.hpp"
#include "poly.hpp"
#include "prover.hpp"
#include <vector>

class Verifier {
    std::mt19937_64 rng;
    double total_time_ms;

    F61 mle_eval(const std::vector<F61> &vals, const std::vector<F61> &point);

public:
    Verifier(uint64_t seed = 0xCAFEBABE) : total_time_ms(0), rng(seed) {}

    struct VerifyResult {
        bool accepted;
        double verifier_time_ms;
        int layers_checked;
        int folds_checked;
    };

    VerifyResult verify(const Prover::ProveResult &proof,
                        const std::vector<WitnessLayer> &witness,
                        const ModelGraph &g);
};
