#pragma once
#include "field.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

enum class LayerType : uint8_t {
    CONV, RELU, AVGPOOL, FC, ADD, FLATTEN
};

struct ConvMeta {
    int c_out, c_in, kh, kw, sh, sw, ph, pw, groups;
};

struct FCMeta {
    int out_features, in_features;
};

struct LayerDesc {
    LayerType type;
    int id;
    int input_ids[2];
    int num_inputs;
    ConvMeta conv;
    FCMeta   fc;
    int pool_k;
    int out_c, out_h, out_w;
};

struct QuantizedTensor {
    std::vector<F61> data;
    int numel;
};

static int64_t float_to_fixed(float v, int frac_bits) {
    return (int64_t)std::round(v * (1 << frac_bits));
}

static F61 fixed_to_field(int64_t x) {
    if (x >= 0) return F61((uint64_t)x);
    return F61(MOD - (uint64_t)(-x));
}

struct ModelGraph {
    std::vector<LayerDesc> layers;
    std::vector<QuantizedTensor> weights;
    std::vector<QuantizedTensor> biases;
    QuantizedTensor input;
    int frac_bits;

    void set_frac_bits(int fb) { frac_bits = fb; }

    QuantizedTensor load_bin(const std::string &path, int expected) {
        QuantizedTensor qt;
        qt.numel = expected;
        qt.data.resize(expected);
        std::ifstream f(path, std::ios::binary);
        std::vector<float> buf(expected);
        f.read(reinterpret_cast<char*>(buf.data()), expected * sizeof(float));
        for (int i = 0; i < expected; ++i)
            qt.data[i] = fixed_to_field(float_to_fixed(buf[i], frac_bits));
        return qt;
    }

    ConvMeta parse_conv_meta(const std::string &path) {
        ConvMeta m{};
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            int val = std::stoi(line.substr(eq + 1));
            if (key == "c_out") m.c_out = val;
            else if (key == "c_in") m.c_in = val;
            else if (key == "kh") m.kh = val;
            else if (key == "kw") m.kw = val;
            else if (key == "sh") m.sh = val;
            else if (key == "sw") m.sw = val;
            else if (key == "ph") m.ph = val;
            else if (key == "pw") m.pw = val;
            else if (key == "groups") m.groups = val;
        }
        return m;
    }

    FCMeta parse_fc_meta(const std::string &path) {
        FCMeta m{};
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            int val = std::stoi(line.substr(eq + 1));
            if (key == "out_features") m.out_features = val;
            else if (key == "in_features") m.in_features = val;
        }
        return m;
    }
};

static void compute_conv_output_shape(const ConvMeta &m, int in_h, int in_w,
                                       int &out_h, int &out_w) {
    out_h = (in_h + 2 * m.ph - m.kh) / m.sh + 1;
    out_w = (in_w + 2 * m.pw - m.kw) / m.sw + 1;
}

static int ceil_log2(uint64_t n) {
    int r = 0;
    uint64_t p = 1;
    while (p < n) { p <<= 1; ++r; }
    return r;
}
