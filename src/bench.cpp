// Capacity / noise / algebra benchmarks for FHRR, GHRR and HeRR.
// Emits newline-delimited JSON so the Python driver can aggregate it.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "vsa/fhrr.hpp"
#include "vsa/ghrr.hpp"
#include "vsa/herr.hpp"

using namespace vsa;

namespace {

struct Args {
  std::map<std::string, std::string> kv;
  bool has(const std::string& k) const { return kv.count(k) > 0; }
  std::string get(const std::string& k, const std::string& d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  double num(const std::string& k, double d = 0.0) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stod(it->second);
  }
  int inum(const std::string& k, int d = 0) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stoi(it->second);
  }
};

Args parse(int argc, char** argv, int from = 1) {
  Args a;
  for (int i = from; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      auto eq = s.find('=');
      if (eq != std::string::npos)
        a.kv[s.substr(2, eq - 2)] = s.substr(eq + 1);
      else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        a.kv[s.substr(2)] = argv[++i];
      else
        a.kv[s.substr(2)] = "1";
    }
  }
  return a;
}

void emit(const std::string& suite, const std::string& system, int D, int m, double diag,
          int M, int trials, double acc, double acc_sd, double margin, int extra_int,
          double extra_dbl, const std::string& note) {
  std::printf(
      "{\"suite\":\"%s\",\"system\":\"%s\",\"D\":%d,\"m\":%d,\"diag\":%g,\"M\":%d,"
      "\"trials\":%d,\"acc\":%.6f,\"acc_sd\":%.6f,\"margin\":%.6f,\"extra_i\":%d,"
      "\"extra_f\":%.6f,\"note\":\"%s\"}\n",
      suite.c_str(), system.c_str(), D, m, diag, M, trials, acc, acc_sd, margin, extra_int,
      extra_dbl, note.c_str());
}

struct TrialResult {
  double acc;
  double margin;
};

// ---------------- FHRR capacity ----------------
TrialResult fhrr_trial(int D, int M, rng_t& rng) {
  std::vector<FHRR> roles, fillers;
  roles.reserve(M);
  fillers.reserve(M);
  for (int i = 0; i < M; ++i) {
    roles.push_back(FHRR::random(D, rng));
    fillers.push_back(FHRR::random(D, rng));
  }
  FHRR mem(D);
  for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
  mem.normalize();

  int ok = 0;
  double margin_sum = 0.0;
  for (int i = 0; i < M; ++i) {
    FHRR est = mem.unbind(roles[i]);
    int arg = 0;
    double bs = -1e300, second = -1e300;
    for (int j = 0; j < M; ++j) {
      double s = est.sim(fillers[j]);
      if (s > bs) {
        second = bs;
        bs = s;
        arg = j;
      } else if (s > second) {
        second = s;
      }
    }
    ok += (arg == i);
    margin_sum += (bs - second);
  }
  return {static_cast<double>(ok) / M, margin_sum / M};
}

// ---------------- GHRR capacity ----------------
TrialResult ghrr_trial(int D, int m, double diag, int M, rng_t& rng) {
  std::vector<GHRR> roles, fillers;
  roles.reserve(M);
  fillers.reserve(M);
  for (int i = 0; i < M; ++i) {
    roles.push_back(GHRR::random_unitary(D, m, diag, rng));
    fillers.push_back(GHRR::random_unitary(D, m, diag, rng));
  }
  GHRR mem(D, m);
  for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
  mem.normalize();

  int ok = 0;
  double margin_sum = 0.0;
  for (int i = 0; i < M; ++i) {
    GHRR est = mem.left_unbind(roles[i]);
    int arg = 0;
    double bs = -1e300, second = -1e300;
    for (int j = 0; j < M; ++j) {
      double s = est.sim(fillers[j]);
      if (s > bs) {
        second = bs;
        bs = s;
        arg = j;
      } else if (s > second) {
        second = s;
      }
    }
    ok += (arg == i);
    margin_sum += (bs - second);
  }
  return {static_cast<double>(ok) / M, margin_sum / M};
}

// ---------------- HeRR capacity ----------------
// roles: distinct random monomial exponents in [0, spread)
// fillers: sparse phase polynomials of length D with `k` non-zeros
TrialResult herr_trial(int D, int M, int k, int role_terms, int spread, bool peel, rng_t& rng,
                       int* support_out = nullptr) {
  std::vector<HeRR> roles, fillers;
  roles.reserve(M);
  fillers.reserve(M);
  std::vector<int> used;
  for (int i = 0; i < M; ++i) fillers.push_back(HeRR::sparse_phase(D, k, 0, rng));
  for (int i = 0; i < M; ++i) {
    HeRR r;
    if (role_terms <= 1) {
      int e;
      bool dup;
      do {
        e = static_cast<int>(rng() % static_cast<uint64_t>(spread));
        dup = false;
        for (int u : used) dup = dup || (u == e);
      } while (dup);
      used.push_back(e);
      r = HeRR::monomial(e);
    } else {
      // Sparse multi-term role, kept distinct by an offset block per pair.
      do {
        r = HeRR::random_sparse_terms(role_terms, spread, rng);
        r.lo += i * 7;
      } while (false);
    }
    roles.push_back(r);
  }
  HeRR mem;
  mem.lo = 0;
  for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
  if (support_out) *support_out = mem.support();

  std::vector<double> sc;
  std::vector<int> dec = HeRR::decode(mem, roles, fillers, &sc, peel);
  int ok = 0;
  for (int i = 0; i < M; ++i) ok += (dec[i] == i);
  double msum = 0.0;
  for (double s : sc) msum += s;
  return {static_cast<double>(ok) / M, msum / M};
}

template <typename Fn>
void run_sweep(const std::string& suite, const std::string& system, int D, int m, double diag,
               const std::vector<int>& Ms, int trials, Fn fn, const std::string& note) {
  for (int M : Ms) {
    std::vector<double> accs, margins;
    for (int t = 0; t < trials; ++t) {
      rng_t rng(0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(t) << 32) ^
                static_cast<uint64_t>(D * 131 + M * 7 + m));
      TrialResult r = fn(rng, M);
      accs.push_back(r.acc);
      margins.push_back(r.margin);
    }
    double mean = 0, sd = 0, mm = 0;
    for (double a : accs) mean += a;
    mean /= accs.size();
    for (double a : accs) sd += (a - mean) * (a - mean);
    sd = std::sqrt(sd / accs.size());
    for (double a : margins) mm += a;
    mm /= margins.size();
    emit(suite, system, D, m, diag, M, trials, mean, sd, mm, 0, 0.0, note);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: vsa-bench <capacity|herr|noise|algebra> [--D=.. --M=.. --trials=..]\n");
    return 2;
  }
  std::string cmd = argv[1];
  Args a = parse(argc, argv, 2);

  if (cmd == "capacity") {
    std::vector<int> Ds;
    for (int D : {64, 128, 256, 512, 1024}) Ds.push_back(D);
    std::vector<int> Ms;
    for (int M : {4, 8, 16, 32, 48, 64, 96, 128, 192}) Ms.push_back(M);
    int trials = a.inum("trials", 10);
    for (int D : Ds) {
      run_sweep("capacity", "fhrr", D, 0, 1.0, Ms, trials,
                [D](rng_t& r, int M) { return fhrr_trial(D, M, r); }, "fhrr");
    }
    std::vector<int> Gm;
    for (int M : {4, 8, 16, 32, 48, 64, 96}) Gm.push_back(M);
    for (int diag_i : {0, 50, 100}) {
      double diag = diag_i / 100.0;
      for (int D : {64, 128, 256, 512}) {
        for (int m : {2, 3, 4}) {
          for (int M : Gm) {
            if (static_cast<long long>(D) * m * m > 8192) continue;  // keep runtime sane
            std::vector<double> accs;
            for (int t = 0; t < trials; ++t) {
              rng_t rng(0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(t) << 32) ^
                        static_cast<uint64_t>(D * 131 + M * 7 + m * 3 + diag_i));
              accs.push_back((M == 0) ? 0.0 : ghrr_trial(D, m, diag, M, rng).acc);
            }
            double mean = 0, sd = 0;
            for (double x : accs) mean += x;
            mean /= accs.size();
            for (double x : accs) sd += (x - mean) * (x - mean);
            sd = std::sqrt(sd / accs.size());
            emit("capacity", "ghrr", D, m, diag, M, trials, mean, sd, 0.0, 0, 0.0, "ghrr");
          }
        }
      }
    }
  } else if (cmd == "herr") {
    std::vector<int> Ms;
    for (int M : {8, 16, 32, 48, 64, 96}) Ms.push_back(M);
    int trials = a.inum("trials", 5);
    int D = a.inum("D", 128);
    for (int k : {8, 16, 32, 64}) {
      for (int peel = 0; peel <= 1; ++peel) {
        for (int M : Ms) {
          std::vector<double> accs;
          int support = 0;
          for (int t = 0; t < trials; ++t) {
            rng_t rng(0x1234567ULL ^ (static_cast<uint64_t>(t) << 32) ^
                      static_cast<uint64_t>(D * 131 + M * 7 + k * 3 + peel));
            TrialResult r = herr_trial(D, M, k, 1, 4096, peel != 0, rng, &support);
            accs.push_back(r.acc);
          }
          double mean = 0, sd = 0;
          for (double x : accs) mean += x;
          mean /= accs.size();
          for (double x : accs) sd += (x - mean) * (x - mean);
          sd = std::sqrt(sd / accs.size());
          emit("herr", peel ? "herr_peel" : "herr_oneshot", D, 1, 0.0, M, trials, mean, sd, 0.0,
               k, static_cast<double>(support), "sparse_filler");
        }
      }
    }
    // Dense fillers: the control that should show no advantage over FHRR.
    for (int M : Ms) {
      std::vector<double> accs;
      for (int t = 0; t < trials; ++t) {
        rng_t rng(0xABCDEFULL ^ (static_cast<uint64_t>(t) << 32) ^
                  static_cast<uint64_t>(D * 131 + M * 7));
        TrialResult r = herr_trial(D, M, D, 1, 4096, true, rng);
        accs.push_back(r.acc);
      }
      double mean = 0, sd = 0;
      for (double x : accs) mean += x;
      mean /= accs.size();
      for (double x : accs) sd += (x - mean) * (x - mean);
      sd = std::sqrt(sd / accs.size());
      emit("herr", "herr_dense_peel", D, 1, 0.0, M, trials, mean, sd, 0.0, D, 0.0, "dense_filler");
    }
  } else if (cmd == "noise") {
    // Additive complex Gaussian noise on a loaded memory, measure decoding.
    int D = a.inum("D", 128);
    int M = a.inum("M", 16);
    int trials = a.inum("trials", 20);
    for (double snr_db : {-6.0, -3.0, 0.0, 3.0, 6.0, 10.0, 20.0, 40.0}) {
      std::vector<double> accs;
      for (int t = 0; t < trials; ++t) {
        rng_t rng(0x77ULL ^ (static_cast<uint64_t>(t) << 32) ^
                  static_cast<uint64_t>(D * 31 + M * 5 + static_cast<int>(snr_db * 10)));
        std::vector<FHRR> roles, fillers;
        for (int i = 0; i < M; ++i) {
          roles.push_back(FHRR::random(D, rng));
          fillers.push_back(FHRR::random(D, rng));
        }
        FHRR mem(D);
        for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
        mem.normalize();
        double p = mem.norm2() / D;
        double sigma = std::sqrt(p / std::pow(10.0, snr_db / 10.0));
        std::normal_distribution<double> nd(0.0, sigma / std::sqrt(2.0));
        for (auto& v : mem.z) v += cd(nd(rng), nd(rng));
        int ok = 0;
        for (int i = 0; i < M; ++i) {
          FHRR est = mem.unbind(roles[i]);
          int arg = 0;
          double bs = -1e300;
          for (int j = 0; j < M; ++j) {
            double s = est.sim(fillers[j]);
            if (s > bs) {
              bs = s;
              arg = j;
            }
          }
          ok += (arg == i);
        }
        accs.push_back(static_cast<double>(ok) / M);
      }
      double mean = 0;
      for (double x : accs) mean += x;
      mean /= accs.size();
      emit("noise", "fhrr", D, 0, 0.0, M, trials, mean, 0.0, 0.0, 0, snr_db, "snr_db");
    }
  } else if (cmd == "algebra") {
    // Machine-checkable algebraic identities.
    rng_t rng(42);
    const int D = 128;
    FHRR a = FHRR::random(D, rng), b = FHRR::random(D, rng);
    double e1 = (a.bind(b)).unbind(b).sim(a);
    std::printf("{\"check\":\"fhrr_unbind_exact\",\"value\":%.12f}\n", e1);
    std::printf("{\"check\":\"fhrr_self\",\"value\":%.12f}\n", a.sim(a));

    GHRR ga = GHRR::random_unitary(32, 3, 0.0, rng), gb = GHRR::random_unitary(32, 3, 0.0, rng);
    std::printf("{\"check\":\"ghrr_unbind_exact\",\"value\":%.12f}\n",
                ga.bind(gb).unbind(gb).sim(ga));
    std::printf("{\"check\":\"ghrr_noncomm\",\"value\":%.12f}\n",
                ga.bind(gb).sim(gb.bind(ga)));
    GHRR da = GHRR::random_unitary(32, 3, 1.0, rng), db = GHRR::random_unitary(32, 3, 1.0, rng);
    std::printf("{\"check\":\"ghrr_diag_comm\",\"value\":%.12f}\n",
                da.bind(db).sim(db.bind(da)));
    GHRR h1 = GHRR::random_unitary(32, 3, 0.5, rng);
    std::printf("{\"check\":\"ghrr_self\",\"value\":%.12f}\n", h1.sim(h1));

    HeRR p = HeRR::dense_phase(16, 0, rng), q = HeRR::monomial(5);
    HeRR pq = p.bind(q);
    std::printf("{\"check\":\"herr_shift_len\",\"value\":%d}\n", pq.support());
    std::printf("{\"check\":\"herr_shift_exact\",\"value\":%.12f}\n",
                pq.shift_scale(-5).sim(p));
    HeRR c1 = HeRR::dense_phase(8, 0, rng), c2 = HeRR::dense_phase(8, 4, rng);
    HeRR prod = c1.bind(c2);
    double err = 0.0;
    for (int e = 0; e < 15; ++e) {
      cd want{0.0, 0.0};
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          if (i + 4 + j == e) want += c1.at(i) * c2.at(4 + j);
      err = std::max(err, std::abs(prod.at(e) - want));
    }
    std::printf("{\"check\":\"herr_conv_error\",\"value\":%.3e}\n", err);

    // Probe: distribution of the similarity of independent GHRR vectors.
    for (int m : {2, 3, 4}) {
      for (int D : {64, 640, 6400}) {
        double mean = 0, sq = 0;
        const int T = 20;
        for (int t = 0; t < T; ++t) {
          GHRR x = GHRR::random_unitary(D, m, 0.0, rng);
          GHRR y = GHRR::random_unitary(D, m, 0.0, rng);
          double s = x.sim(y);
          mean += s;
          sq += s * s;
        }
        mean /= T;
        sq = std::sqrt(sq / T - mean * mean);
        std::printf("{\"check\":\"ghrr_indep\",\"m\":%d,\"D\":%d,\"mean\":%.5f,\"sd\":%.5f}\n", m,
                    D, mean, sq);
      }
    }
  } else {
    std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    return 2;
  }
  return 0;
}
