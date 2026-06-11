#pragma once
#include "field.hpp"
#include "poly.hpp"
#include "graph.hpp"
#include <vector>
#include <chrono>

struct WitnessLayer {
    std::vector<F61> vals;
    int bit_length;
    uint64_t size;
};

struct SumcheckTranscript {
    std::vector<QuadPoly> round_polys;
    std::vector<F61> challenges;
    F61 final_claim;
    uint64_t byte_size;
};

struct ResidualFoldingState {
    F61 claim_main;
    F61 claim_skip;
    F61 alpha;
    F61 folded;
};

class Prover {
    std::vector<WitnessLayer> witness;
    std::vector<SumcheckTranscript> transcripts;
    double total_time_ms;
    uint64_t total_proof_bytes;
    std::mt19937_64 rng;

    void execute_conv(int layer_id, const ModelGraph &g);
    void execute_relu(int layer_id, const ModelGraph &g);
    void execute_avgpool(int layer_id, const ModelGraph &g);
    void execute_fc(int layer_id, const ModelGraph &g);
    void execute_add(int layer_id, const ModelGraph &g);
    void execute_flatten(int layer_id, const ModelGraph &g);

    F61 mle_eval(const std::vector<F61> &vals, const std::vector<F61> &point);

    SumcheckTranscript prove_layer(int layer_id, const std::vector<F61> &r_out,
                                    const ModelGraph &g);

public:
    Prover(uint64_t seed = 0xDEADBEEF) : total_time_ms(0), total_proof_bytes(0), rng(seed) {}

    void compute_witness(const ModelGraph &g);

    struct ProveResult {
        std::vector<SumcheckTranscript> transcripts;
        std::vector<ResidualFoldingState> fold_states;
        std::vector<std::vector<F61>> challenges_per_layer;
        double prover_time_ms;
        uint64_t proof_size_bytes;
    };

    ProveResult prove(const ModelGraph &g);
    const std::vector<WitnessLayer> &get_witness() const { return witness; }
};
