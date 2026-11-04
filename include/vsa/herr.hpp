// HeRR (Helical Reduced Representations).
//
// Hypervectors are Laurent polynomials  P(z) = sum_k c_k z^k  with exponents in
// Z (no wrap-around). Binding of two vectors is polynomial multiplication, so
// the exponent set is closed under addition and the algebra is exact — this is
// the "helical" escape from the group-ring capacity floor of FHRR.  The cost is
// paid in support growth: convolution of two supports of size n costs 2n-1
// coefficients, so HeRR trades a hard algebraic capacity ceiling for a third
// budget: memory / numerical precision.
//
// Decoding uses the mean-and-peel scheme: correlate the memory with the known
// role, take the strongest match, subtract it, and repeat.
#pragma once

#include "vsa/common.hpp"

namespace vsa {

struct HeRR {
  int lo = 0;              // exponent of c[0]
  std::vector<cd> c;       // dense coefficients, c[i] <-> exponent lo + i

  HeRR() = default;

  int hi() const { return lo + static_cast<int>(c.size()) - 1; }
  int support() const { return static_cast<int>(c.size()); }
  int n_coeffs() const { return static_cast<int>(c.size()); }

  static HeRR monomial(int exponent, cd coeff = cd(1.0, 0.0)) {
    HeRR v;
    v.lo = exponent;
    v.c = {coeff};
    return v;
  }

  // Dense block of `n` unit-modulus coefficients starting at exponent `start`.
  static HeRR dense_phase(int n, int start, rng_t& rng) {
    HeRR v;
    v.lo = start;
    v.c.resize(n);
    std::uniform_real_distribution<double> ud(0.0, 2.0 * kPi);
    for (int i = 0; i < n; ++i) v.c[i] = std::polar(1.0, ud(rng));
    return v;
  }

  // `k` non-zero unit-modulus coefficients on a random support inside [start, start+n).
  static HeRR sparse_phase(int n, int k, int start, rng_t& rng) {
    HeRR v;
    v.lo = start;
    v.c.assign(n, cd(0.0, 0.0));
    std::uniform_real_distribution<double> ud(0.0, 2.0 * kPi);
    std::uniform_int_distribution<int> pos(0, n - 1);
    int placed = 0;
    while (placed < k) {
      int p = pos(rng);
      if (std::abs(v.c[p]) > 0.0) continue;
      v.c[p] = std::polar(1.0, ud(rng));
      ++placed;
    }
    return v;
  }

  static HeRR random_sparse_terms(int k_terms, int spread, rng_t& rng) {
    if (k_terms <= 0) k_terms = 1;
    HeRR v;
    std::vector<std::pair<int, cd>> terms;
    std::uniform_int_distribution<int> pos(0, spread - 1);
    std::uniform_real_distribution<double> ud(0.0, 2.0 * kPi);
    std::vector<int> used;
    for (int t = 0; t < k_terms; ++t) {
      int e;
      bool dup;
      do {
        e = pos(rng);
        dup = false;
        for (int u : used) dup = dup || (u == e);
      } while (dup);
      used.push_back(e);
      terms.emplace_back(e, std::polar(1.0, ud(rng)));
    }
    std::sort(terms.begin(), terms.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    v.lo = terms.front().first;
    v.c.resize(terms.back().first - v.lo + 1, cd(0.0, 0.0));
    for (auto& t : terms) v.c[t.first - v.lo] = t.second;
    return v;
  }

  // Polynomial multiplication (Laurent): exponents add, supports convolve.
  HeRR bind(const HeRR& o) const {
    HeRR r;
    r.lo = lo + o.lo;
    r.c.assign(c.size() + o.c.size() - 1, cd(0.0, 0.0));
    for (size_t i = 0; i < c.size(); ++i) {
      if (c[i] == cd(0.0, 0.0)) continue;
      for (size_t j = 0; j < o.c.size(); ++j) r.c[i + j] += c[i] * o.c[j];
    }
    // Trim exact zeros at the edges to keep the support honest.
    while (r.c.size() > 1 && r.c.front() == cd(0.0, 0.0)) {
      r.c.erase(r.c.begin());
      ++r.lo;
    }
    while (r.c.size() > 1 && r.c.back() == cd(0.0, 0.0)) r.c.pop_back();
    return r;
  }

  // Multiply by z^shift * coeff — exact and cheap (the HeRR "unbind" primitive
  // for monomial roles).
  HeRR shift_scale(int shift, cd coeff = cd(1.0, 0.0)) const {
    HeRR r = *this;
    r.lo += shift;
    for (auto& v : r.c) v *= coeff;
    return r;
  }

  void add_scaled(const HeRR& o, double w) {
    int nlo = std::min(lo, o.lo);
    int nhi = std::max(hi(), o.hi());
    std::vector<cd> nc(static_cast<size_t>(nhi - nlo + 1), cd(0.0, 0.0));
    for (size_t i = 0; i < c.size(); ++i) nc[(lo + static_cast<int>(i)) - nlo] += c[i];
    for (size_t i = 0; i < o.c.size(); ++i) nc[(o.lo + static_cast<int>(i)) - nlo] += w * o.c[i];
    int nlo2 = nlo, nhi2 = nhi;
    while (nhi2 > nlo2 && nc.front() == cd(0.0, 0.0)) {
      nc.erase(nc.begin());
      ++nlo2;
    }
    while (nhi2 > nlo2 && nc.back() == cd(0.0, 0.0)) {
      nc.pop_back();
      --nhi2;
    }
    lo = nlo2;
    c = std::move(nc);
  }

  // Inner product <a, b> = sum_k conj(a_k) b_k over the aligned exponent range.
  cd inner(const HeRR& o) const {
    int nlo = std::max(lo, o.lo);
    int nhi = std::min(hi(), o.hi());
    cd s{0.0, 0.0};
    for (int e = nlo; e <= nhi; ++e) s += std::conj(at(e)) * o.at(e);
    return s;
  }

  // <this, coeff * z^shift * o> without allocating the shifted polynomial.
  inline cd inner_shifted(const HeRR& o, int shift, cd coeff) const {
    int olo = o.lo + shift, ohi = o.hi() + shift;
    int nlo = std::max(lo, olo);
    int nhi = std::min(hi(), ohi);
    cd s{0.0, 0.0};
    const int off = shift - lo;
    const int n = static_cast<int>(c.size());
    for (int e = nlo; e <= nhi; ++e) {
      int oi = e - shift - o.lo;
      int mi = e - lo;
      if (mi < 0 || mi >= n) continue;
      s += std::conj(c[static_cast<size_t>(mi)]) * (coeff * o.c[static_cast<size_t>(oi)]);
    }
    (void)off;
    return s;
  }

  // Real cosine similarity on the aligned window, normalised by the geometric
  // mean of the two energies inside that window.
  double sim(const HeRR& o) const {
    int nlo = std::max(lo, o.lo);
    int nhi = std::min(hi(), o.hi());
    if (nhi < nlo) return 0.0;
    double num = 0.0, ea = 0.0, eb = 0.0;
    for (int e = nlo; e <= nhi; ++e) {
      cd x = at(e), y = o.at(e);
      num += (std::conj(x) * y).real();
      ea += std::norm(x);
      eb += std::norm(y);
    }
    double den = std::sqrt(ea * eb);
    return den > 1e-12 ? num / den : 0.0;
  }

  double energy() const {
    double s = 0.0;
    for (const auto& v : c) s += std::norm(v);
    return s;
  }

  double norm() const { return std::sqrt(energy()); }

  void scale(double s) {
    for (auto& v : c) v *= s;
  }

  void normalize() {
    double n = norm();
    if (n > 1e-12) scale(1.0 / n);
  }

  cd at(int exponent) const {
    int i = exponent - lo;
    if (i < 0 || i >= static_cast<int>(c.size())) return cd(0.0, 0.0);
    return c[static_cast<size_t>(i)];
  }

  void set(int exponent, cd value) {
    if (c.empty()) {
      lo = exponent;
      c = {value};
      return;
    }
    if (exponent < lo) {
      c.insert(c.begin(), static_cast<size_t>(lo - exponent), cd(0.0, 0.0));
      lo = exponent;
    }
    int i = exponent - lo;
    if (i >= static_cast<int>(c.size())) c.resize(static_cast<size_t>(i) + 1, cd(0.0, 0.0));
    c[static_cast<size_t>(i)] = value;
  }

  // ---- Mean-and-peel decoder ------------------------------------------------
  //
  // memory      : the superposition  M = sum_i bind(role_i, filler_i)
  // roles       : known role polynomials (one per stored pair)
  // fillers     : fixed vocabulary the decoded filler is matched against
  // returns     : for every role, the index of the decoded filler in `fillers`
  //
  // peel == false -> plain one-shot correlation (the FHRR-equivalent baseline)
  // peel == true  -> iterative: decode strongest, subtract, repeat.
  //
  // Monomial roles (the common case: z^k) take an allocation-free fast path.
  static std::vector<int> decode(const HeRR& memory,
                                 const std::vector<HeRR>& roles,
                                 const std::vector<HeRR>& fillers,
                                 std::vector<double>* scores = nullptr,
                                 bool peel = true,
                                 int max_peel = -1) {
    const int R = static_cast<int>(roles.size());
    const int F = static_cast<int>(fillers.size());
    std::vector<int> out(R, 0);
    std::vector<double> best(R, 0.0);
    if (R == 0 || F == 0) {
      if (scores) *scores = best;
      return out;
    }
    if (max_peel < 0 || max_peel > R) max_peel = R;

    std::vector<double> fnorm(F, 1.0);
    for (int f = 0; f < F; ++f) {
      double n = fillers[f].norm();
      fnorm[f] = n > 1e-12 ? n : 1.0;
    }
    std::vector<double> rnorm(R, 1.0);
    std::vector<char> mono(R, 0);
    std::vector<int> rshift(R, 0);
    std::vector<cd> rcoef(R, cd(1.0, 0.0));
    for (int r = 0; r < R; ++r) {
      double n = roles[r].norm();
      rnorm[r] = n > 1e-12 ? n : 1.0;
      if (roles[r].support() == 1) {
        mono[r] = 1;
        rshift[r] = roles[r].lo;
        rcoef[r] = roles[r].c[0] / std::abs(roles[r].c[0]);
      }
    }

    auto score_pair = [&](const HeRR& src, int r, int f) -> double {
      const double den = rnorm[r] * fnorm[f];
      if (mono[r]) return src.inner_shifted(fillers[f], rshift[r], rcoef[r]).real() / den;
      return src.inner(roles[r].bind(fillers[f])).real() / den;
    };

    if (!peel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int r = 0; r < R; ++r) {
        double bs = -1e300;
        int bf = 0;
        for (int f = 0; f < F; ++f) {
          double sc = score_pair(memory, r, f);
          if (sc > bs) {
            bs = sc;
            bf = f;
          }
        }
        out[r] = bf;
        best[r] = bs;
      }
      if (scores) *scores = best;
      return out;
    }

    std::vector<char> done(R, 0);
    HeRR residual = memory;
    for (int iter = 0; iter < max_peel; ++iter) {
      int best_role = -1, best_filler = -1;
      double best_score = -1e300;
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int lr = -1, lf = -1;
        double ls = -1e300;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int r = 0; r < R; ++r) {
          if (done[r]) continue;
          for (int f = 0; f < F; ++f) {
            double sc = score_pair(residual, r, f);
            if (sc > ls) {
              ls = sc;
              lr = r;
              lf = f;
            }
          }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          if (ls > best_score) {
            best_score = ls;
            best_role = lr;
            best_filler = lf;
          }
        }
      }
      if (best_role < 0) break;
      out[best_role] = best_filler;
      best[best_role] = best_score;
      done[best_role] = 1;
      // Subtract the decoded pair with the matched amplitude.
      const double den = rnorm[best_role] * fnorm[best_filler];
      cd amp = residual.inner(roles[best_role].bind(fillers[best_filler])) / den;
      HeRR comp = roles[best_role].bind(fillers[best_filler]);
      comp.scale(amp.real());
      residual.add_scaled(comp, -1.0);
    }
    if (scores) *scores = best;
    return out;
  }
};

}  // namespace vsa
