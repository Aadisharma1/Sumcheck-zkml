#include "prover.hpp"
#include "verifier.hpp"
#include "graph.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

static void build_resnet18_graph(ModelGraph &g, const std::string &weight_dir) {
    g.set_frac_bits(16);

    std::vector<std::string> tags;
    for (auto &entry : fs::directory_iterator(weight_dir)) {
        std::string name = entry.path().filename().string();
        if (name.find("_meta.txt") != std::string::npos) {
            std::string tag = name.substr(0, name.find("_meta.txt"));
            if (tag == "input") continue;
            tags.push_back(tag);
        }
    }
    std::sort(tags.begin(), tags.end());

    std::cerr << "found " << tags.size() << " weight tags\n";
    for (auto &t : tags) std::cerr << "  " << t << "\n";

    if (tags.size() < 21) {
        std::cerr << "FATAL: expected 21 weight tags for ResNet-18, got " << tags.size() << "\n";
        return;
    }

    int lid = 0;
    auto add_conv = [&](int in_id, const std::string &tag, int in_h, int in_w) -> int {
        std::string meta_path = weight_dir + "/" + tag + "_meta.txt";
        std::string w_path    = weight_dir + "/" + tag + "_weight.bin";
        std::string b_path    = weight_dir + "/" + tag + "_bias.bin";

        ConvMeta cm = g.parse_conv_meta(meta_path);
        int oh, ow;
        compute_conv_output_shape(cm, in_h, in_w, oh, ow);

        LayerDesc d{};
        d.type = LayerType::CONV;
        d.id = lid;
        d.input_ids[0] = in_id;
        d.num_inputs = 1;
        d.conv = cm;
        d.out_c = cm.c_out; d.out_h = oh; d.out_w = ow;

        g.layers.push_back(d);
        g.weights.resize(lid + 1);
        g.biases.resize(lid + 1);
        g.weights[lid] = g.load_bin(w_path, cm.c_out * (cm.c_in / cm.groups) * cm.kh * cm.kw);
        g.biases[lid] = g.load_bin(b_path, cm.c_out);

        return lid++;
    };

    auto add_relu = [&](int in_id) -> int {
        LayerDesc d{};
        d.type = LayerType::RELU;
        d.id = lid;
        d.input_ids[0] = in_id;
        d.num_inputs = 1;
        auto &prev = g.layers[in_id];
        d.out_c = prev.out_c; d.out_h = prev.out_h; d.out_w = prev.out_w;
        g.layers.push_back(d);
        g.weights.resize(lid + 1);
        g.biases.resize(lid + 1);
        return lid++;
    };

    auto add_add = [&](int a, int b_id) -> int {
        LayerDesc d{};
        d.type = LayerType::ADD;
        d.id = lid;
        d.input_ids[0] = a;
        d.input_ids[1] = b_id;
        d.num_inputs = 2;
        auto &prev = g.layers[a];
        d.out_c = prev.out_c; d.out_h = prev.out_h; d.out_w = prev.out_w;
        g.layers.push_back(d);
        g.weights.resize(lid + 1);
        g.biases.resize(lid + 1);
        return lid++;
    };

    auto add_avgpool = [&](int in_id, int k) -> int {
        LayerDesc d{};
        d.type = LayerType::AVGPOOL;
        d.id = lid;
        d.input_ids[0] = in_id;
        d.num_inputs = 1;
        d.pool_k = k;
        auto &prev = g.layers[in_id];
        d.out_c = prev.out_c;
        d.out_h = k > 0 ? prev.out_h / k : 1;
        d.out_w = k > 0 ? prev.out_w / k : 1;
        if (d.out_h == 0) d.out_h = 1;
        if (d.out_w == 0) d.out_w = 1;
        g.layers.push_back(d);
        g.weights.resize(lid + 1);
        g.biases.resize(lid + 1);
        return lid++;
    };

    auto add_flatten = [&](int in_id) -> int {
        LayerDesc d{};
        d.type = LayerType::FLATTEN;
        d.id = lid;
        d.input_ids[0] = in_id;
        d.num_inputs = 1;
        auto &prev = g.layers[in_id];
        d.out_c = prev.out_c * prev.out_h * prev.out_w;
        d.out_h = 1; d.out_w = 1;
        g.layers.push_back(d);
        g.weights.resize(lid + 1);
        g.biases.resize(lid + 1);
        return lid++;
    };

    auto add_fc = [&](int in_id, const std::string &tag) -> int {
        std::string meta_path = weight_dir + "/" + tag + "_meta.txt";
        std::string w_path    = weight_dir + "/" + tag + "_weight.bin";
        std::string b_path    = weight_dir + "/" + tag + "_bias.bin";

        FCMeta fm = g.parse_fc_meta(meta_path);

        LayerDesc d{};
        d.type = LayerType::FC;
        d.id = lid;
        d.input_ids[0] = in_id;
        d.num_inputs = 1;
        d.fc = fm;
        d.out_c = fm.out_features; d.out_h = 1; d.out_w = 1;

        g.layers.push_back(d);
        g.weights.resize(lid + 1);
        g.biases.resize(lid + 1);
        g.weights[lid] = g.load_bin(w_path, fm.out_features * fm.in_features);
        g.biases[lid] = g.load_bin(b_path, fm.out_features);

        return lid++;
    };

    // layer 0: input placeholder
    {
        LayerDesc d{};
        d.type = LayerType::FLATTEN;
        d.id = 0;
        d.input_ids[0] = -1;
        d.num_inputs = 0;
        d.out_c = 3; d.out_h = 32; d.out_w = 32;
        g.layers.push_back(d);
        g.weights.resize(1);
        g.biases.resize(1);
        lid = 1;
    }

    int cur = 0;
    int idx = 0;

    // tags[0] = 000_conv1
    int c0 = add_conv(cur, tags[idx++], 32, 32);
    int r0 = add_relu(c0);
    cur = r0;

    // tags[1..20]: 8 basic blocks + FC
    // layer1: blocks 0,1 (no downsample)
    // layer2: block 0 (downsample), block 1
    // layer3: block 0 (downsample), block 1
    // layer4: block 0 (downsample), block 1
    auto basic_block = [&](int in_id, bool has_ds) -> int {
        auto &pin = g.layers[in_id];
        int c1 = add_conv(in_id, tags[idx++], pin.out_h, pin.out_w);
        int r1 = add_relu(c1);
        auto &pr1 = g.layers[r1];
        int c2 = add_conv(r1, tags[idx++], pr1.out_h, pr1.out_w);

        int skip = in_id;
        if (has_ds) {
            skip = add_conv(in_id, tags[idx++], pin.out_h, pin.out_w);
        }

        int a = add_add(c2, skip);
        return add_relu(a);
    };

    cur = basic_block(cur, false);  // layer1.0
    cur = basic_block(cur, false);  // layer1.1
    cur = basic_block(cur, true);   // layer2.0 (ds)
    cur = basic_block(cur, false);  // layer2.1
    cur = basic_block(cur, true);   // layer3.0 (ds)
    cur = basic_block(cur, false);  // layer3.1
    cur = basic_block(cur, true);   // layer4.0 (ds)
    cur = basic_block(cur, false);  // layer4.1

    // idx should now be 20 (tags[20] = 020_fc)
    auto &plast = g.layers[cur];
    int pool = add_avgpool(cur, plast.out_h);
    int flat = add_flatten(pool);
    add_fc(flat, tags[idx]);
}

int main(int argc, char **argv) {
    std::string weight_dir = "fused_weights";
    if (argc > 1) weight_dir = argv[1];

    std::string input_path = weight_dir + "/input.bin";

    ModelGraph g;
    g.set_frac_bits(16);

    {
        std::ifstream meta(weight_dir + "/input_meta.txt");
        std::string line;
        int total = 3 * 32 * 32;
        if (std::getline(meta, line)) {
            auto eq = line.find('=');
            if (eq != std::string::npos) {
                std::string shape = line.substr(eq + 1);
                int prod = 1;
                std::stringstream ss(shape);
                std::string tok;
                while (std::getline(ss, tok, 'x'))
                    prod *= std::stoi(tok);
                total = prod;
            }
        }
        g.input = g.load_bin(input_path, total);
    }

    std::cerr << "building resnet18 graph...\n";
    build_resnet18_graph(g, weight_dir);
    std::cerr << "graph built: " << g.layers.size() << " layers\n";

    Prover prover;
    std::cerr << "computing witness...\n";
    prover.compute_witness(g);

    auto &w = prover.get_witness();
    std::cerr << "output layer size: " << w.back().size << "\n";
    if (w.back().size >= 10) {
        std::cerr << "logits: ";
        for (int i = 0; i < 10 && i < (int)w.back().vals.size(); ++i)
            std::cerr << w.back().vals[i].v << " ";
        std::cerr << "\n";
    }

    std::cerr << "running prover...\n";
    auto proof = prover.prove(g);

    std::cerr << "=== PROVER ===\n";
    std::cerr << "prover_time_ms = " << proof.prover_time_ms << "\n";
    std::cerr << "proof_size_bytes = " << proof.proof_size_bytes << "\n";
    std::cerr << "transcripts = " << proof.transcripts.size() << "\n";
    std::cerr << "fold_states = " << proof.fold_states.size() << "\n";

    Verifier verifier;
    std::cerr << "running verifier...\n";
    auto vr = verifier.verify(proof, w, g);

    std::cerr << "=== VERIFIER ===\n";
    std::cerr << "accepted = " << (vr.accepted ? "YES" : "NO") << "\n";
    std::cerr << "verifier_time_ms = " << vr.verifier_time_ms << "\n";
    std::cerr << "layers_checked = " << vr.layers_checked << "\n";
    std::cerr << "folds_checked = " << vr.folds_checked << "\n";

    std::cout << "RESULT,"
              << proof.prover_time_ms << ","
              << vr.verifier_time_ms << ","
              << proof.proof_size_bytes << ","
              << (vr.accepted ? "ACCEPT" : "REJECT") << "\n";

    return vr.accepted ? 0 : 1;
}
