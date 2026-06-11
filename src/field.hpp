#pragma once
#include <cstdint>
#include <cstring>
#include <random>
#include <iostream>

static constexpr uint64_t MOD = (1ULL << 61) - 1;

#ifdef _MSC_VER
#include <intrin.h>
static inline uint64_t mul_mod_mersenne(uint64_t a, uint64_t b) {
    uint64_t hi, lo;
    lo = _umul128(a, b, &hi);
    uint64_t lo61 = lo & MOD;
    uint64_t carry = (lo >> 61) | (hi << 3);
    uint64_t s = lo61 + carry;
    return s >= MOD ? s - MOD : s;
}
#else
static inline uint64_t mul_mod_mersenne(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)(r & MOD);
    uint64_t hi = (uint64_t)(r >> 61);
    uint64_t s = lo + hi;
    return s >= MOD ? s - MOD : s;
}
#endif

struct F61 {
    uint64_t v;

    F61() : v(0) {}
    explicit F61(uint64_t x) : v(x % MOD) {}

    static F61 one()  { return F61(1); }
    static F61 zero() { return F61(0); }

    static F61 rand(std::mt19937_64 &rng) {
        return F61(rng() % MOD);
    }

    F61 operator+(const F61 &o) const {
        uint64_t s = v + o.v;
        return F61(s >= MOD ? s - MOD : s);
    }

    F61 operator-(const F61 &o) const {
        return F61(v >= o.v ? v - o.v : MOD - o.v + v);
    }

    F61 operator*(const F61 &o) const {
        F61 r;
        r.v = mul_mod_mersenne(v, o.v);
        return r;
    }

    F61 &operator+=(const F61 &o) { *this = *this + o; return *this; }
    F61 &operator-=(const F61 &o) { *this = *this - o; return *this; }
    F61 &operator*=(const F61 &o) { *this = *this * o; return *this; }

    bool operator==(const F61 &o) const { return v == o.v; }
    bool operator!=(const F61 &o) const { return v != o.v; }

    F61 inv() const {
        int64_t a = v, b = MOD, x0 = 1, x1 = 0;
        while (b) {
            int64_t q = a / b;
            int64_t t = b; b = a - q * b; a = t;
            t = x1; x1 = x0 - q * x1; x0 = t;
        }
        return F61((x0 % (int64_t)MOD + MOD) % MOD);
    }

    void clear() { v = 0; }
    bool is_zero() const { return v == 0; }
};
