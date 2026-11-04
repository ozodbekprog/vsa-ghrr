// Common types for the VSA kernel library (FHRR / GHRR / HeRR).
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace vsa {

using cd = std::complex<double>;
using rng_t = std::mt19937_64;

constexpr double kPi = 3.14159265358979323846;

inline double clamp_unit(double x) { return std::max(-1.0, std::min(1.0, x)); }

// Complex m x m matrix helpers, row-major, m is small (2..8).
inline void mat_mul(const cd* A, const cd* B, cd* C, int m) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      cd s{0.0, 0.0};
      for (int k = 0; k < m; ++k) s += A[i * m + k] * B[k * m + j];
      C[i * m + j] = s;
    }
  }
}

// C = A * B^H  (the "unbind" product)
inline void mat_mul_adj(const cd* A, const cd* B, cd* C, int m) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      cd s{0.0, 0.0};
      for (int k = 0; k < m; ++k) s += A[i * m + k] * std::conj(B[j * m + k]);
      C[i * m + j] = s;
    }
  }
}

// C = A^H * B  (left multiplication by the adjoint)
inline void mat_mul_left_adj(const cd* A, const cd* B, cd* C, int m) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      cd s{0.0, 0.0};
      for (int k = 0; k < m; ++k) s += std::conj(A[k * m + i]) * B[k * m + j];
      C[i * m + j] = s;
    }
  }
}

// Haar-ish random unitary by complex Gram-Schmidt (columns orthonormal).
inline void random_unitary(cd* Q, int m, rng_t& rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) Q[i * m + j] = cd(nd(rng), nd(rng));
  for (int j = 0; j < m; ++j) {
    for (int k = 0; k < j; ++k) {
      cd dot{0.0, 0.0};
      for (int i = 0; i < m; ++i) dot += std::conj(Q[i * m + k]) * Q[i * m + j];
      for (int i = 0; i < m; ++i) Q[i * m + j] -= dot * Q[i * m + k];
    }
    double nrm = 0.0;
    for (int i = 0; i < m; ++i) nrm += std::norm(Q[i * m + j]);
    nrm = std::sqrt(nrm);
    if (nrm < 1e-12) {  // degenerate draw, restart column
      for (int i = 0; i < m; ++i) Q[i * m + j] = (i == j) ? cd(1.0, 0.0) : cd(0.0, 0.0);
      continue;
    }
    for (int i = 0; i < m; ++i) Q[i * m + j] /= nrm;
    // NB: do NOT phase-fix the diagonal here.  Forcing Q_jj real gives the
    // matrix a non-zero mean, which biases sim() between *independent* vectors
    // to ~1/m.  Plain Haar keeps E[Tr(A B^H)] = 0.
  }
}

}  // namespace vsa
