#include "prover.hpp"
#include <algorithm>
#include <numeric>

void Prover::compute_witness(const ModelGraph &g) {
    witness.resize(g.layers.size());
    witness[0].vals = g.input.data;
    witness[0].size = g.input.numel;
    witness[0].bit_length = ceil_log2(witness[0].size);

    for (size_t l = 1; l < g.layers.size(); ++l) {
        auto &desc = g.layers[l];
        switch (desc.type) {
            case LayerType::CONV:    execute_conv(l, g);    break;
            case LayerType::RELU:    execute_relu(l, g);    break;
            case LayerType::AVGPOOL: execute_avgpool(l, g); break;
            case LayerType::FC:      execute_fc(l, g);      break;
            case LayerType::ADD:     execute_add(l, g);     break;
            case LayerType::FLATTEN: execute_flatten(l, g); break;
        }
        witness[l].size = witness[l].vals.size();
        witness[l].bit_length = ceil_log2(witness[l].size);
    }
}

void Prover::execute_conv(int lid, const ModelGraph &g) {
    auto &desc = g.layers[lid];
    auto &cm = desc.conv;
    auto &in = witness[desc.input_ids[0]].vals;
    int in_c = cm.c_in, in_h = desc.out_h, in_w = desc.out_w;

    int prev_l = desc.input_ids[0];
    auto &prev = g.layers[prev_l];
    int p_h = prev.out_h, p_w = prev.out_w;

    int oh, ow;
    compute_conv_output_shape(cm, p_h, p_w, oh, ow);

    auto &W = g.weights[lid].data;
    auto &B = g.biases[lid].data;

    int out_sz = cm.c_out * oh * ow;
    witness[lid].vals.resize(out_sz);
    auto &out = witness[lid].vals;

    F61 rescale = F61(1ULL << g.frac_bits).inv();

    for (int oc = 0; oc < cm.c_out; ++oc) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                F61 acc = B[oc];
                for (int ic = 0; ic < cm.c_in; ++ic) {
                    for (int ky = 0; ky < cm.kh; ++ky) {
                        for (int kx = 0; kx < cm.kw; ++kx) {
                            int iy = oy * cm.sh - cm.ph + ky;
                            int ix = ox * cm.sw - cm.pw + kx;
                            if (iy < 0 || iy >= p_h || ix < 0 || ix >= p_w) continue;
                            int w_idx = oc * (cm.c_in * cm.kh * cm.kw) + ic * (cm.kh * cm.kw) + ky * cm.kw + kx;
                            int i_idx = ic * (p_h * p_w) + iy * p_w + ix;
                            acc += W[w_idx] * in[i_idx] * rescale;
                        }
                    }
                }
                out[oc * (oh * ow) + oy * ow + ox] = acc;
            }
        }
    }
}

void Prover::execute_relu(int lid, const ModelGraph &g) {
    auto &in = witness[g.layers[lid].input_ids[0]].vals;
    witness[lid].vals.resize(in.size());
    F61 half_mod((MOD + 1) / 2);
    for (size_t i = 0; i < in.size(); ++i)
        witness[lid].vals[i] = (in[i].v < half_mod.v) ? in[i] : F61::zero();
}

void Prover::execute_avgpool(int lid, const ModelGraph &g) {
    auto &desc = g.layers[lid];
    int prev_l = desc.input_ids[0];
    auto &in = witness[prev_l].vals;
    auto &prev = g.layers[prev_l];
    int c = prev.out_c, h = prev.out_h, w = prev.out_w;
    int k = desc.pool_k ? desc.pool_k : h;

    int oh = h / k, ow = w / k;
    if (oh == 0) oh = 1;
    if (ow == 0) ow = 1;

    F61 inv_k2 = F61((uint64_t)(k * k)).inv();
    witness[lid].vals.resize(c * oh * ow);
    auto &out = witness[lid].vals;

    for (int ci = 0; ci < c; ++ci) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                F61 acc = F61::zero();
                for (int py = 0; py < k; ++py)
                    for (int px = 0; px < k; ++px) {
                        int iy = oy * k + py, ix = ox * k + px;
                        if (iy < h && ix < w)
                            acc += in[ci * h * w + iy * w + ix];
                    }
                out[ci * oh * ow + oy * ow + ox] = acc * inv_k2;
            }
        }
    }
}

void Prover::execute_fc(int lid, const ModelGraph &g) {
    auto &desc = g.layers[lid];
    auto &fm = desc.fc;
    auto &in = witness[desc.input_ids[0]].vals;
    auto &W = g.weights[lid].data;
    auto &B = g.biases[lid].data;

    F61 rescale = F61(1ULL << g.frac_bits).inv();
    witness[lid].vals.resize(fm.out_features);
    auto &out = witness[lid].vals;

    for (int o = 0; o < fm.out_features; ++o) {
        F61 acc = B[o];
        for (int i = 0; i < fm.in_features; ++i)
            acc += W[o * fm.in_features + i] * in[i] * rescale;
        out[o] = acc;
    }
}

void Prover::execute_add(int lid, const ModelGraph &g) {
    auto &a = witness[g.layers[lid].input_ids[0]].vals;
    auto &b = witness[g.layers[lid].input_ids[1]].vals;
    size_t n = std::min(a.size(), b.size());
    witness[lid].vals.resize(n);
    for (size_t i = 0; i < n; ++i)
        witness[lid].vals[i] = a[i] + b[i];
}

void Prover::execute_flatten(int lid, const ModelGraph &g) {
    witness[lid].vals = witness[g.layers[lid].input_ids[0]].vals;
}

F61 Prover::mle_eval(const std::vector<F61> &vals, const std::vector<F61> &point) {
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

SumcheckTranscript Prover::prove_layer(int lid, const std::vector<F61> &r_out,
                                         const ModelGraph &g) {
    SumcheckTranscript tx;
    auto &out_vals = witness[lid].vals;
    int bl = witness[lid].bit_length;
    uint64_t N = 1ULL << bl;

    std::vector<F61> eq_table;
    init_beta_table(eq_table, bl, r_out.data(), F61::one());

    auto &desc = g.layers[lid];
    int in_id = desc.input_ids[0];
    auto &in_vals = witness[in_id].vals;
    int in_bl = witness[in_id].bit_length;

    int total_vars = in_bl;
    std::vector<F61> bookkeeping(1ULL << total_vars, F61::zero());
    for (uint64_t i = 0; i < std::min((uint64_t)in_vals.size(), (uint64_t)(1ULL << total_vars)); ++i)
        bookkeeping[i] = in_vals[i];

    std::vector<F61> mult_table(1ULL << total_vars, F61::zero());
    for (uint64_t u = 0; u < N && u < (uint64_t)(1ULL << total_vars); ++u)
        mult_table[u] = eq_table[u % N];

    F61 claim = F61::zero();
    for (uint64_t i = 0; i < (uint64_t)(1ULL << total_vars); ++i)
        claim += bookkeeping[i] * mult_table[i];

    tx.byte_size = 0;
    for (int j = 0; j < total_vars; ++j) {
        uint64_t half = 1ULL << (total_vars - j - 1);
        F61 s0, s1, s2;

        for (uint64_t i = 0; i < half; ++i) {
            F61 bk0 = bookkeeping[2 * i], bk1 = bookkeeping[2 * i + 1];
            F61 mt0 = mult_table[2 * i],  mt1 = mult_table[2 * i + 1];

            s0 += bk0 * mt0;
            s1 += bk1 * mt1;
            F61 bk2 = bk1 + bk1 - bk0;
            F61 mt2 = mt1 + mt1 - mt0;
            s2 += bk2 * mt2;
        }

        QuadPoly poly(s2 + s0 - s1 - s1, s1 - s0, s0);
        tx.round_polys.push_back(poly);
        tx.byte_size += 3 * 8;

        F61 r_j = F61::rand(rng);
        tx.challenges.push_back(r_j);

        for (uint64_t i = 0; i < half; ++i) {
            bookkeeping[i] = bookkeeping[2 * i] * (F61::one() - r_j) + bookkeeping[2 * i + 1] * r_j;
            mult_table[i]  = mult_table[2 * i]  * (F61::one() - r_j) + mult_table[2 * i + 1]  * r_j;
        }
    }

    tx.final_claim = bookkeeping[0] * mult_table[0];
    tx.byte_size += 8;
    return tx;
}

Prover::ProveResult Prover::prove(const ModelGraph &g) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    ProveResult result;
    result.proof_size_bytes = 0;

    int n_layers = g.layers.size();
    std::vector<std::vector<F61>> r_points(n_layers);

    {
        int last = n_layers - 1;
        int bl = witness[last].bit_length;
        r_points[last].resize(bl);
        for (int i = 0; i < bl; ++i)
            r_points[last][i] = F61::rand(rng);
    }

    for (int l = n_layers - 1; l >= 1; --l) {
        auto &desc = g.layers[l];

        if (desc.type == LayerType::ADD) {
            int id_a = desc.input_ids[0], id_b = desc.input_ids[1];
            int bl = witness[l].bit_length;

            F61 v1 = mle_eval(witness[id_a].vals, r_points[l]);
            F61 v2 = mle_eval(witness[id_b].vals, r_points[l]);

            ResidualFoldingState fold;
            fold.claim_main = v1;
            fold.claim_skip = v2;
            fold.alpha = F61::rand(rng);
            fold.folded = v1 + fold.alpha * v2;
            result.fold_states.push_back(fold);

            r_points[id_a] = r_points[l];
            r_points[id_b] = r_points[l];

            result.proof_size_bytes += 3 * 8;
            continue;
        }

        if (desc.type == LayerType::FLATTEN || desc.type == LayerType::RELU) {
            int in_id = desc.input_ids[0];
            r_points[in_id] = r_points[l];
            continue;
        }

        auto tx = prove_layer(l, r_points[l], g);
        result.proof_size_bytes += tx.byte_size;
        result.transcripts.push_back(std::move(tx));

        int in_id = desc.input_ids[0];
        auto &last_tx = result.transcripts.back();
        r_points[in_id] = last_tx.challenges;
        if ((int)r_points[in_id].size() < witness[in_id].bit_length) {
            r_points[in_id].resize(witness[in_id].bit_length);
            for (int i = last_tx.challenges.size(); i < witness[in_id].bit_length; ++i)
                r_points[in_id][i] = F61::rand(rng);
        }

        result.challenges_per_layer.push_back(r_points[in_id]);
    }

    auto t1 = clock::now();
    result.prover_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}
