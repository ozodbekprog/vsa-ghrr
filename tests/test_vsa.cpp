// Unit tests for the VSA algebra. No external test framework on purpose.
#include <cstdio>
#include <string>
#include <vector>

#include "vsa/fhrr.hpp"
#include "vsa/ghrr.hpp"
#include "vsa/herr.hpp"

using namespace vsa;

static int g_fail = 0;
static int g_pass = 0;

static void check(const std::string& name, bool ok, double value = 0.0) {
  if (ok) {
    ++g_pass;
    std::printf("  ok   %-42s %g\n", name.c_str(), value);
  } else {
    ++g_fail;
    std::printf("  FAIL %-42s %g\n", name.c_str(), value);
  }
}

static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

int main() {
  rng_t rng(20261104);

  std::printf("[FHRR]\n");
  {
    FHRR a = FHRR::random(256, rng), b = FHRR::random(256, rng);
    double e = a.bind(b).unbind(b).sim(a);
    check("bind/unbind is exact", near(e, 1.0, 1e-10), e);
    check("self similarity is 1", near(a.sim(a), 1.0, 1e-12), a.sim(a));
    check("independent similarity ~ 0", std::abs(a.sim(b)) < 0.15, a.sim(b));
    FHRR m(256);
    for (int i = 0; i < 4; ++i) m.add_scaled(FHRR::random(256, rng), 1.0);
    m.normalize();
    check("bundle normalised", near(m.norm2() / 256, 1.0, 1e-9), m.norm2() / 256);
  }

  std::printf("[GHRR]\n");
  for (int m : {2, 3, 4}) {
    for (double diag : {0.0, 0.5, 1.0}) {
      GHRR a = GHRR::random_unitary(64, m, diag, rng);
      GHRR b = GHRR::random_unitary(64, m, diag, rng);
      double e = a.bind(b).unbind(b).sim(a);
      check("m=" + std::to_string(m) + " diag=" + std::to_string(diag) + " unbind exact",
            near(e, 1.0, 1e-9), e);
      double l = a.bind(b).left_unbind(a).sim(b);
      check("m=" + std::to_string(m) + " diag=" + std::to_string(diag) + " left_unbind exact",
            near(l, 1.0, 1e-9), l);
      double un = a.bind(b).sim(b.bind(a));
      if (diag == 0.0) {
        check("m=" + std::to_string(m) + " non-commutative", un < 0.9, un);
      } else if (diag == 1.0) {
        check("m=" + std::to_string(m) + " commutative (diag)", near(un, 1.0, 1e-9), un);
      }
    }
  }
  {
    // Verify unitarity of generated blocks: A A^H = I.
    GHRR a = GHRR::random_unitary(8, 4, 0.0, rng);
    double worst = 0.0;
    const size_t cs = a.comp_size();
    for (int j = 0; j < a.D; ++j) {
      cd prod[16], tmp[16];
      mat_mul_adj(a.a.data() + j * cs, a.a.data() + j * cs, tmp, 4);
      for (int i = 0; i < 16; ++i) prod[i] = tmp[i];
      for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) {
          cd want = (i == k) ? cd(1, 0) : cd(0, 0);
          worst = std::max(worst, std::abs(prod[i * 4 + k] - want));
        }
    }
    check("random blocks are unitary", worst < 1e-10, worst);
  }
  {
    GHRR a = GHRR::random_unitary(64, 3, 0.0, rng);
    check("ghrr self similarity is 1", near(a.sim(a), 1.0, 1e-12), a.sim(a));
    GHRR b = GHRR::random_unitary(64, 3, 0.0, rng);
    check("ghrr independent ~ 0", std::abs(a.sim(b)) < 0.1, a.sim(b));
  }

  std::printf("[HeRR]\n");
  {
    HeRR a = HeRR::dense_phase(16, -3, rng);
    check("dense support", a.support() == 16 && a.lo == -3, a.support());
    HeRR s = a.shift_scale(5);
    check("shift keeps support", s.support() == 16 && s.lo == 2, s.lo);
    check("shift is invertible", near(s.shift_scale(-5).sim(a), 1.0, 1e-12),
          s.shift_scale(-5).sim(a));
  }
  {
    // Convolution against a naive O(n^2) reference.
    HeRR a = HeRR::dense_phase(9, 0, rng);
    HeRR b = HeRR::dense_phase(7, 4, rng);
    HeRR p = a.bind(b);
    double worst = 0.0;
    for (int e = 4; e <= 4 + 6 + 8; ++e) {
      cd want{0, 0};
      for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 7; ++j)
          if (i + 4 + j == e) want += a.at(i) * b.at(4 + j);
      worst = std::max(worst, std::abs(p.at(e) - want));
    }
    check("laurent convolution correct", worst < 1e-12, worst);
    check("support grows to n+m-1", p.support() == 9 + 7 - 1, p.support());
  }
  {
    // Exact decoding at low load, both one-shot and peel.
    for (bool peel : {false, true}) {
      const int M = 16, D = 128, k = 8;
      std::vector<HeRR> roles, fillers;
      std::vector<int> used;
      for (int i = 0; i < M; ++i) fillers.push_back(HeRR::sparse_phase(D, k, 0, rng));
      for (int i = 0; i < M; ++i) {
        int e;
        bool dup;
        do {
          e = static_cast<int>(rng() % 4096);
          dup = false;
          for (int u : used) dup = dup || (u == e);
        } while (dup);
        used.push_back(e);
        roles.push_back(HeRR::monomial(e));
      }
      HeRR mem;
      mem.lo = 0;
      for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
      std::vector<int> dec = HeRR::decode(mem, roles, fillers, nullptr, peel);
      int ok = 0;
      for (int i = 0; i < M; ++i) ok += (dec[i] == i);
      check(peel ? "peel decode 16/16" : "one-shot decode 16/16", ok == M, ok);
    }
  }
  {
    // Unbinding a stored pair must beat every non-matching role.
    HeRR f = HeRR::sparse_phase(64, 8, 0, rng);
    HeRR r1 = HeRR::monomial(11), r2 = HeRR::monomial(29);
    HeRR mem = r1.bind(f);
    mem.add_scaled(r2.bind(HeRR::sparse_phase(64, 8, 0, rng)), 1.0);
    double rec = mem.inner_shifted(f, 11, cd(1, 0)).real() / (f.norm() * f.norm());
    check("single-pair recovery dominates crosstalk", rec > 0.9, rec);
  }

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
