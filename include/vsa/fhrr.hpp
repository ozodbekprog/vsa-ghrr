// FHRR (Fourier Holographic Reduced Representations) — the baseline.
// Bind = elementwise complex product, unbind = product with the conjugate.
// Capacity is limited by the O(D / log D) crosstalk floor of the group ring.
#pragma once

#include "vsa/common.hpp"

namespace vsa {

struct FHRR {
  int D = 0;
  std::vector<cd> z;  // unit-modulus phases

  FHRR() = default;
  explicit FHRR(int d) : D(d), z(static_cast<size_t>(d), cd(0.0, 0.0)) {}

  static FHRR random(int d, rng_t& rng) {
    FHRR v(d);
    std::uniform_real_distribution<double> ud(0.0, 2.0 * kPi);
    for (int i = 0; i < d; ++i) v.z[i] = std::polar(1.0, ud(rng));
    return v;
  }

  static FHRR ones(int d) {
    FHRR v(d);
    for (int i = 0; i < d; ++i) v.z[i] = cd(1.0, 0.0);
    return v;
  }

  FHRR bind(const FHRR& o) const {
    FHRR r(D);
    for (int i = 0; i < D; ++i) r.z[i] = z[i] * o.z[i];
    return r;
  }

  FHRR unbind(const FHRR& o) const {
    FHRR r(D);
    for (int i = 0; i < D; ++i) r.z[i] = z[i] * std::conj(o.z[i]);
    return r;
  }

  void add_scaled(const FHRR& o, double w) {
    for (int i = 0; i < D; ++i) z[i] += w * o.z[i];
  }

  // Re(<a, b>) / D  — 1.0 for identical unit-modulus vectors.
  double sim(const FHRR& o) const {
    double s = 0.0;
    for (int i = 0; i < D; ++i) s += (z[i] * std::conj(o.z[i])).real();
    return s / static_cast<double>(D);
  }

  double norm2() const {
    double s = 0.0;
    for (int i = 0; i < D; ++i) s += std::norm(z[i]);
    return s;
  }

  void normalize() {
    double n = std::sqrt(norm2() / static_cast<double>(D));
    if (n > 1e-12)
      for (int i = 0; i < D; ++i) z[i] /= n;
  }

  int n_coeffs() const { return D; }
};

}  // namespace vsa
