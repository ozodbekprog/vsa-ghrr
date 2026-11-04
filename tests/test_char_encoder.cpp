// Unit tests for vsa::CharEncoder (FHRR random indexing of word spellings).
// No external test framework on purpose — same shape as tests/test_vsa.cpp.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "vsa/char_encoder.hpp"

using namespace vsa;
using cf = std::complex<float>;

static int g_fail = 0;
static int g_pass = 0;

static void check(const std::string& name, bool ok, double value = 0.0) {
  if (ok) {
    ++g_pass;
    std::printf("  ok   %-46s %g\n", name.c_str(), value);
  } else {
    ++g_fail;
    std::printf("  FAIL %-46s %g\n", name.c_str(), value);
  }
}

static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// a..z -> 1..26, space -> 27 (0 is padding / "no character").
static std::vector<int> to_ids(const std::string& s) {
  std::vector<int> ids;
  ids.reserve(s.size());
  for (char ch : s) {
    if (ch == ' ') ids.push_back(27);
    else if (ch >= 'a' && ch <= 'z') ids.push_back(ch - 'a' + 1);
    else ids.push_back(28);  // anything else collapses to one "other" symbol
  }
  return ids;
}

static double cosine(const std::vector<cf>& a, const std::vector<cf>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>((a[i] * std::conj(b[i])).real());
    na += static_cast<double>(std::norm(a[i]));
    nb += static_cast<double>(std::norm(b[i]));
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0;
  return dot / std::sqrt(na * nb);
}

static double norm(const std::vector<cf>& a) {
  double s = 0.0;
  for (const cf& z : a) s += static_cast<double>(std::norm(z));
  return std::sqrt(s);
}

int main() {
  const int V = 64, D = 512, L = 32;

  std::printf("[CharEncoder] vocab=%d D=%d max_len=%d\n", V, D, L);
  {
    CharEncoder enc(V, D, L, 20261104);

    // Codebooks must be unit modulus and fixed in size.
    {
      double worst = 0.0;
      for (const cf& z : enc.char_vecs) worst = std::max(worst, std::abs(std::abs(z) - 1.0));
      for (const cf& z : enc.pos_vecs) worst = std::max(worst, std::abs(std::abs(z) - 1.0));
      check("codebooks are unit modulus",
            worst < 1e-6 && enc.char_vecs.size() == static_cast<size_t>(V) * D &&
                enc.pos_vecs.size() == static_cast<size_t>(L) * D,
            worst);
    }

    std::vector<cf> olma = enc.encode(to_ids("olma"));
    check("encode dimension == D", olma.size() == static_cast<size_t>(D), olma.size());
    check("encode is unit norm", near(norm(olma), 1.0, 1e-5), norm(olma));
    check("self cosine == 1", near(cosine(olma, olma), 1.0, 1e-5), cosine(olma, olma));

    // Empty input -> all zeros.
    {
      std::vector<cf> e = enc.encode({});
      double peak = 0.0;
      for (const cf& z : e) peak = std::max(peak, static_cast<double>(std::abs(z)));
      check("empty input is all zero", peak == 0.0 && e.size() == static_cast<size_t>(D), peak);
    }

    // Length matters: trailing space is a different word.
    {
      double c = cosine(olma, enc.encode(to_ids("olma ")));
      check("'olma' vs 'olma ' differs (len matters)", c < 0.98 && c < 1.0, c);
    }

    // Truncation at max_len.
    {
      std::vector<int> long_ids = to_ids("abcdefghijklmnopqrstuvwxyzabcdefghijklmn");
      std::vector<int> cut(long_ids.begin(), long_ids.begin() + L);
      double c = cosine(enc.encode(long_ids), enc.encode(cut));
      check("ids longer than max_len are truncated", near(c, 1.0, 1e-6), c);
    }
  }

  std::printf("[anagram separation, several seeds]\n");
  {
    double worst = -1.0;
    for (uint64_t seed : {1ull, 2ull, 3ull, 7ull, 20261104ull, 999331ull}) {
      CharEncoder enc(V, D, L, seed);
      double c = cosine(enc.encode(to_ids("olma")), enc.encode(to_ids("amlo")));
      worst = std::max(worst, c);
      check("cos(olma, amlo) < 0.2  seed=" + std::to_string(seed), c < 0.2, c);
    }
    std::printf("  worst-case cos(olma, amlo) over seeds: %g\n", worst);
  }

  std::printf("[encode_batch]\n");
  {
    CharEncoder enc(V, D, L, 20261104);
    const std::vector<std::string> words = {"olma", "amlo", "olma ", "suv", "maktab", ""};
    const int n = static_cast<int>(words.size());
    std::vector<int> ids(static_cast<size_t>(n) * L, 0);
    std::vector<int> lens(n, 0);
    for (int w = 0; w < n; ++w) {
      const std::vector<int> row = to_ids(words[w]);
      lens[w] = static_cast<int>(std::min<size_t>(row.size(), L));
      for (int i = 0; i < lens[w]; ++i)
        ids[static_cast<size_t>(w) * L + i] = row[static_cast<size_t>(i)];
    }
    std::vector<cf> out(static_cast<size_t>(n) * D, cf(0.0f, 0.0f));
    enc.encode_batch(ids.data(), lens.data(), n, out.data());

    double worst = 0.0;
    for (int w = 0; w < n; ++w) {
      std::vector<cf> ref = enc.encode(to_ids(words[w]));
      for (int j = 0; j < D; ++j)
        worst = std::max(
            worst, static_cast<double>(std::abs(out[static_cast<size_t>(w) * D + j] - ref[j])));
    }
    check("encode_batch == encode row-by-row", worst < 1e-6, worst);

    // Padding beyond lens[w] must be ignored.
    {
      std::vector<int> padded = ids;
      for (int w = 0; w < n; ++w)
        for (int i = lens[w]; i < L; ++i) padded[static_cast<size_t>(w) * L + i] = 13;
      std::vector<cf> out2(static_cast<size_t>(n) * D, cf(0.0f, 0.0f));
      enc.encode_batch(padded.data(), lens.data(), n, out2.data());
      double d = 0.0;
      for (size_t i = 0; i < out.size(); ++i)
        d = std::max(d, static_cast<double>(std::abs(out[i] - out2[i])));
      check("padding beyond lens is ignored", d < 1e-6, d);
    }

    // n_words == 0 must be a no-op.
    enc.encode_batch(nullptr, nullptr, 0, nullptr);
    check("encode_batch(0 words) is a no-op", true, 0.0);
  }

  std::printf("[determinism]\n");
  {
    CharEncoder a(V, D, L, 42), b(V, D, L, 42), c(V, D, L, 43);
    bool same_codebook = (a.char_vecs == b.char_vecs) && (a.pos_vecs == b.pos_vecs);
    std::vector<cf> ea = a.encode(to_ids("maktab")), eb = b.encode(to_ids("maktab"));
    bool same_vec = (ea == eb);
    check("same seed -> identical codebooks", same_codebook, 0.0);
    check("same seed -> identical encoding", same_vec, cosine(ea, eb));
    double diff = cosine(a.encode(to_ids("maktab")), c.encode(to_ids("maktab")));
    check("different seed -> different encoding", std::abs(diff) < 0.5, diff);
  }

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
