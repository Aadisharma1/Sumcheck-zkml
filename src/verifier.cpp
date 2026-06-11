#include "verifier.hpp"
#include <chrono>

F61 Verifier::mle_eval(const std::vector<F61> &vals, const std::vector<F61> &point) {
    int n = point.size();
    uint64_t len = 1ULL << n;
    std::vector<F61> tmp(len, F61::zero());
    for (uint64_t i = 0; i < std::min((uint64_t)vals.size(), len); ++i)
        tmp[i] = vals[i];
    for (int i = 0; i < n; ++i) {
        uint64_t half = len >> 1;
        for (uint64_t j = 0; j < half; ++j)
            tmp[j] = tmp[2 * j] * (F61::one() - point[i]) + tmp[2 * j + 1] * point[i];
        len = half;
    }
    return tmp[0];
}

Verifier::VerifyResult Verifier::verify(const Prover::ProveResult &proof,
                                         const std::vector<WitnessLayer> &witness,
                                         const ModelGraph &g) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    VerifyResult res;
    res.accepted = true;
    res.layers_checked = 0;
    res.folds_checked = 0;

    int n_layers = g.layers.size();
    std::vector<std::vector<F61>> r_points(n_layers);

    {
        int last = n_layers - 1;
        int bl = witness[last].bit_length;
        r_points[last].resize(bl);
        std::mt19937_64 vrng(0xDEADBEEF);
        for (int i = 0; i < bl; ++i)
            r_points[last][i] = F61::rand(vrng);
    }

    F61 current_claim = mle_eval(witness[n_layers - 1].vals, r_points[n_layers - 1]);

    int tx_idx = 0;
    int fold_idx = 0;

    for (int l = n_layers - 1; l >= 1; --l) {
        auto &desc = g.layers[l];

        if (desc.type == LayerType::ADD) {
            int id_a = desc.input_ids[0], id_b = desc.input_ids[1];

            F61 v1 = mle_eval(witness[id_a].vals, r_points[l]);
            F61 v2 = mle_eval(witness[id_b].vals, r_points[l]);

            F61 sum_check = v1 + v2;
            F61 expected = mle_eval(witness[l].vals, r_points[l]);
            if (sum_check != expected) {
                res.accepted = false;
                break;
            }

            if (fold_idx < (int)proof.fold_states.size()) {
                auto &fs = proof.fold_states[fold_idx];
                if (fs.claim_main != v1 || fs.claim_skip != v2) {
                    res.accepted = false;
                    break;
                }
                F61 recomputed = fs.claim_main + fs.alpha * fs.claim_skip;
                if (recomputed != fs.folded) {
                    res.accepted = false;
                    break;
                }
                ++fold_idx;
            }

            r_points[id_a] = r_points[l];
            r_points[id_b] = r_points[l];
            ++res.folds_checked;
            continue;
        }

        if (desc.type == LayerType::FLATTEN || desc.type == LayerType::RELU) {
            r_points[desc.input_ids[0]] = r_points[l];
            continue;
        }

        if (tx_idx >= (int)proof.transcripts.size()) {
            res.accepted = false;
            break;
        }

        auto &tx = proof.transcripts[tx_idx];
        int num_rounds = tx.round_polys.size();

        F61 running = current_claim;
        for (int j = 0; j < num_rounds; ++j) {
            auto &poly = tx.round_polys[j];
            F61 check = poly.eval(F61::zero()) + poly.eval(F61::one());
            if (check != running) {
                res.accepted = false;
                break;
            }
            running = poly.eval(tx.challenges[j]);
        }
        if (!res.accepted) break;

        if (running != tx.final_claim) {
            res.accepted = false;
            break;
        }

        int in_id = desc.input_ids[0];
        r_points[in_id] = tx.challenges;
        if ((int)r_points[in_id].size() < witness[in_id].bit_length) {
            r_points[in_id].resize(witness[in_id].bit_length);
            std::mt19937_64 vrng2(0xDEADBEEF + l);
            for (int i = tx.challenges.size(); i < witness[in_id].bit_length; ++i)
                r_points[in_id][i] = F61::rand(vrng2);
        }

        current_claim = mle_eval(witness[in_id].vals, r_points[in_id]);
        ++tx_idx;
        ++res.layers_checked;
    }

    auto t1 = clock::now();
    res.verifier_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}
