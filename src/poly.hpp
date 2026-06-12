#pragma once
#include "field.hpp"
#include <vector>

struct QuadPoly {
    F61 a, b, c;
    QuadPoly() {}
    QuadPoly(F61 a_, F61 b_, F61 c_) : a(a_), b(b_), c(c_) {}

    F61 eval(const F61 &x) const {
        return a * x * x + b * x + c;
    }

    QuadPoly operator+(const QuadPoly &o) const {
        return {a + o.a, b + o.b, c + o.c};
    }

    QuadPoly &operator+=(const QuadPoly &o) {
        a += o.a; b += o.b; c += o.c;
        return *this;
    }
};
// cubic poly, might cause issues, have to check.
struct CubicPoly {
    F61 a, b, c, d;
    CubicPoly() {}
    CubicPoly(F61 a_, F61 b_, F61 c_, F61 d_) : a(a_), b(b_), c(c_), d(d_) {}

    F61 eval(const F61 &x) const {
        return ((a * x + b) * x + c) * x + d;
    }

    CubicPoly operator+(const CubicPoly &o) const {
        return {a + o.a, b + o.b, c + o.c, d + o.d};
    }
};

struct LinearPoly {
    F61 slope, intercept;
    LinearPoly() {}
    LinearPoly(F61 s, F61 i) : slope(s), intercept(i) {}

    F61 eval(const F61 &x) const { return slope * x + intercept; }

    LinearPoly operator*(const LinearPoly &o) const {
        return {slope * o.intercept + intercept * o.slope,
                intercept * o.intercept};
    }
};

inline LinearPoly interpolate_linear(const F61 &v0, const F61 &v1) {
    return {v1 - v0, v0};
}

inline void init_beta_table(std::vector<F61> &beta, int n, const F61 *r, F61 init) {
    uint64_t len = 1ULL << n;
    beta.resize(len, F61::zero());
    beta[0] = init;
    for (int i = 0; i < n; ++i) {
        uint64_t half = 1ULL << i;
        for (uint64_t j = half; j > 0; --j) {
            uint64_t idx = j - 1;
            beta[2 * idx + 1] = beta[idx] * r[i];
            beta[2 * idx]     = beta[idx] * (F61::one() - r[i]);
        }
    }
}
