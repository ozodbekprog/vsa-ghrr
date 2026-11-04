// GHRR (Generalized Holographic Reduced Representations).
//
// A GHRR hypervector is a sequence of D components, each an m x m complex
// matrix. Binding is the elementwise matrix product, unbinding multiplies by
// the conjugate transpose.  The `diagonality` knob in [0,1] sets the fraction
// of components drawn as diagonal unitaries (which reduces GHRR to FHRR-like
// commutative behaviour); the rest are full unitaries, which makes binding
// non-commutative and therefore order-sensitive.
#pragma once

#include "vsa/common.hpp"

namespace vsa {

struct GHRR {
  int D = 0;  // number of components
  int m = 0;  // matrix side
  std::vector<cd> a;  // D * m * m, row-major

  GHRR() = default;
  GHRR(int d, int mm)
      : D(d), m(mm), a(static_cast<size_t>(d) * mm * mm, cd(0.0, 0.0)) {}

  size_t comp_size() const { return static_cast<size_t>(m) * m; }
  int n_coeffs() const { return D * m * m; }

  static GHRR random_unitary(int d, int mm, double diagonality, rng_t& rng) {
    GHRR v(d, mm);
    std::uniform_real_distribution<double> ud(0.0, 2.0 * kPi);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    const size_t cs = v.comp_size();
    for (int j = 0; j < d; ++j) {
      cd* blk = v.a.data() + j * cs;
      if (coin(rng) < diagonality) {
        for (int i = 0; i < mm; ++i)
          for (int k = 0; k < mm; ++k) blk[i * mm + k] = (i == k) ? std::polar(1.0, ud(rng)) : cd(0, 0);
      } else {
        vsa::random_unitary(blk, mm, rng);
      }
    }
    return v;
  }

  static GHRR identity(int d, int mm) {
    GHRR v(d, mm);
    const size_t cs = v.comp_size();
    for (int j = 0; j < d; ++j)
      for (int i = 0; i < mm; ++i) v.a[j * cs + i * mm + i] = cd(1.0, 0.0);
    return v;
  }

  // Elementwise matrix product.
  GHRR bind(const GHRR& o) const {
    GHRR r(D, m);
    const size_t cs = comp_size();
    for (int j = 0; j < D; ++j)
      mat_mul(a.data() + j * cs, o.a.data() + j * cs, r.a.data() + j * cs, m);
    return r;
  }

  // Elementwise A_j * B_j^H  (right unbind: exact inverse of bind when B is unitary).
  GHRR unbind(const GHRR& o) const {
    GHRR r(D, m);
    const size_t cs = comp_size();
    for (int j = 0; j < D; ++j)
      mat_mul_adj(a.data() + j * cs, o.a.data() + j * cs, r.a.data() + j * cs, m);
    return r;
  }

  // Elementwise B_j^H * A_j  (left unbind).  This is the decoder that matches
  // bind(role, filler) = role * filler, because matrices do not commute:
  //   role^H * (role * filler) = filler       (correct)
  //   (role * filler) * role^H = role*filler*role^H  (conjugated, NOT filler)
  GHRR left_unbind(const GHRR& o) const {
    GHRR r(D, m);
    const size_t cs = comp_size();
    cd tmp[64];
    for (int j = 0; j < D; ++j) {
      mat_mul_left_adj(o.a.data() + j * cs, a.data() + j * cs, tmp, m);  // o_j^H * A_j
      for (size_t i = 0; i < cs; ++i) r.a[j * cs + i] = tmp[i];
    }
    return r;
  }

  void add_scaled(const GHRR& o, double w) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += w * o.a[i];
  }

  // Re sum_j Tr(A_j B_j^H) / (D * m)  — 1.0 for identical unitaries.
  double sim(const GHRR& o) const {
    double s = 0.0;
    const size_t cs = comp_size();
    for (int j = 0; j < D; ++j) {
      const cd* A = a.data() + j * cs;
      const cd* B = o.a.data() + j * cs;
      cd tr{0.0, 0.0};
      for (int i = 0; i < m; ++i)
        for (int k = 0; k < m; ++k) tr += A[i * m + k] * std::conj(B[i * m + k]);
      s += tr.real();
    }
    return s / (static_cast<double>(D) * m);
  }

  double fro_norm2() const {
    double s = 0.0;
    for (const auto& v : a) s += std::norm(v);
    return s;
  }

  void normalize() {
    const double target = static_cast<double>(D) * m;  // a single unitary has this norm^2
    const double n2 = fro_norm2();
    if (n2 > 1e-12) {
      const double s = std::sqrt(target / n2);
      for (auto& v : a) v *= s;
    }
  }

  // Restrict to the diagonal of every component and return an FHRR phase vector.
  FHRR diagonal_part() const {
    FHRR f(D);
    const size_t cs = comp_size();
    for (int j = 0; j < D; ++j) f.z[j] = a[j * cs + 0];
    return f;
  }
};

}  // namespace vsa
