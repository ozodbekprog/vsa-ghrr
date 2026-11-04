// CharEncoder — FHRR random-indexing encoder for word spellings.
//
// A word is a sum of position-bound character phases:
//   encode(c_0..c_{n-1}) = normalize( sum_i char_vec[c_i] * pos_vec[i] )
// Both codebooks are fixed unit-modulus phase vectors drawn from a seeded
// mt19937_64, so encodings are reproducible and no two characters alias.
// The sum is a bundle: order information lives only in the phase binding, so
// an anagram ("olma" vs "amlo") lands near-orthogonal rather than identical.
// Cost is O(len * D) per word, which is why the batch path writes straight
// into caller-owned memory.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vsa/common.hpp"

namespace vsa {

struct CharEncoder {
  int vocab, D, max_len;
  std::vector<std::complex<float>> char_vecs;  // vocab x D, unit modulus, fixed
  std::vector<std::complex<float>> pos_vecs;   // max_len x D, unit modulus, fixed

  static constexpr float kTwoPi = 6.28318530717958647692f;

  CharEncoder(int vocab, int D, int max_len, uint64_t seed)
      : vocab(vocab), D(D), max_len(max_len) {
    if (vocab <= 0) throw std::invalid_argument("CharEncoder: vocab must be > 0");
    if (D <= 0) throw std::invalid_argument("CharEncoder: D must be > 0");
    if (max_len <= 0) throw std::invalid_argument("CharEncoder: max_len must be > 0");

    rng_t rng(seed);
    char_vecs.resize(static_cast<size_t>(vocab) * static_cast<size_t>(D));
    pos_vecs.resize(static_cast<size_t>(max_len) * static_cast<size_t>(D));
    for (auto& z : char_vecs) z = random_phase(rng);
    for (auto& z : pos_vecs) z = random_phase(rng);
  }

  // sum_i char_vecs[id_i] * pos_vecs[i], then L2-normalise to unit norm.
  // ids longer than max_len are truncated; empty input returns zeros.
  // Ids outside [0, vocab) contribute nothing (treated as padding).
  std::vector<std::complex<float>> encode(const std::vector<int>& ids) const {
    std::vector<std::complex<float>> out(static_cast<size_t>(D), std::complex<float>(0.0f, 0.0f));
    int n = static_cast<int>(ids.size());
    if (n > max_len) n = max_len;
    accumulate(ids.data(), n, out.data());
    normalize_l2(out.data());
    return out;
  }

  // Batched, no allocation: ids is n_words x max_len row-major with lens[n]
  // valid entries per row; out is n_words x D. OpenMP over words if available.
  // `stride` overrides the row length of `ids` (default: this->max_len); the
  // number of positions actually accumulated is min(lens[w], max_len).
  void encode_batch(const int* ids, const int* lens, int n_words, std::complex<float>* out,
                    int stride = -1) const {
    if (n_words <= 0) return;
    if (ids == nullptr || lens == nullptr || out == nullptr)
      throw std::invalid_argument("CharEncoder::encode_batch: null buffer");
    if (stride <= 0) stride = max_len;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int w = 0; w < n_words; ++w) {
      std::complex<float>* o = out + static_cast<size_t>(w) * static_cast<size_t>(D);
      int n = lens[w];
      if (n < 0) n = 0;
      if (n > max_len) n = max_len;
      accumulate(ids + static_cast<size_t>(w) * static_cast<size_t>(stride), n, o);
      normalize_l2(o);
    }
  }

 private:
  // Deterministic uniform phase: 53 bits of mt19937_64 -> u in [0, 1).
  // Uses the raw engine output rather than uniform_real_distribution so the
  // codebooks do not depend on the standard library implementation.
  static std::complex<float> random_phase(rng_t& rng) {
    const double u = static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
    return std::polar(1.0f, kTwoPi * static_cast<float>(u));
  }

  // out += sum_{i<n} char_vecs[ids[i]] * pos_vecs[i]; out is zeroed first.
  // This is the hot loop: one complex multiply-add per component.
  void accumulate(const int* ids, int n, std::complex<float>* out) const {
    for (int j = 0; j < D; ++j) out[j] = std::complex<float>(0.0f, 0.0f);
    for (int i = 0; i < n; ++i) {
      const int id = ids[i];
      if (id < 0 || id >= vocab) continue;
      const std::complex<float>* cv =
          char_vecs.data() + static_cast<size_t>(id) * static_cast<size_t>(D);
      const std::complex<float>* pv =
          pos_vecs.data() + static_cast<size_t>(i) * static_cast<size_t>(D);
      for (int j = 0; j < D; ++j) out[j] += cv[j] * pv[j];
    }
  }

  void normalize_l2(std::complex<float>* v) const {
    float s = 0.0f;
    for (int j = 0; j < D; ++j) s += std::norm(v[j]);
    const float nrm = std::sqrt(s);
    if (nrm <= 1e-12f) return;  // empty / all-padding row stays zero
    const float inv = 1.0f / nrm;
    for (int j = 0; j < D; ++j) v[j] *= inv;
  }
};

}  // namespace vsa
