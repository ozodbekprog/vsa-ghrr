// pybind11 bindings for the VSA kernel library.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "vsa/char_encoder.hpp"
#include "vsa/fhrr.hpp"
#include "vsa/ghrr.hpp"
#include "vsa/herr.hpp"

namespace py = pybind11;
using namespace vsa;

namespace {

py::array_t<std::complex<double>> fhrr_array(const FHRR& v) {
  py::array_t<std::complex<double>> out(v.D);
  auto r = out.mutable_unchecked<1>();
  for (int i = 0; i < v.D; ++i) r(i) = v.z[i];
  return out;
}

FHRR fhrr_from_array(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> a) {
  auto buf = a.request();
  if (buf.ndim != 1) throw std::runtime_error("expected 1-D complex array");
  FHRR v(static_cast<int>(buf.shape[0]));
  const auto* p = static_cast<const std::complex<double>*>(buf.ptr);
  for (int i = 0; i < v.D; ++i) v.z[i] = p[i];
  return v;
}

py::array_t<std::complex<double>> ghrr_array(const GHRR& v) {
  py::array_t<std::complex<double>> out({v.D, v.m, v.m});
  auto r = out.mutable_unchecked<3>();
  const size_t cs = v.comp_size();
  for (int j = 0; j < v.D; ++j)
    for (int i = 0; i < v.m; ++i)
      for (int k = 0; k < v.m; ++k) r(j, i, k) = v.a[j * cs + i * v.m + k];
  return out;
}

py::array_t<std::complex<double>> herr_array(const HeRR& v) {
  py::array_t<std::complex<double>> out(v.support());
  auto r = out.mutable_unchecked<1>();
  for (int i = 0; i < v.support(); ++i) r(i) = v.c[static_cast<size_t>(i)];
  return out;
}

py::array_t<std::complex<float>> char_encode_array(const CharEncoder& enc,
                                                   const std::vector<int>& ids) {
  const std::vector<std::complex<float>> v = enc.encode(ids);
  py::array_t<std::complex<float>> out(v.size());
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

py::array_t<std::complex<float>> char_encode_batch_array(
    const CharEncoder& enc, const std::vector<std::vector<int>>& batch) {
  const int n = static_cast<int>(batch.size());
  py::array_t<std::complex<float>> out({n, enc.D});
  std::vector<int> ids(static_cast<size_t>(n) * enc.max_len, 0);
  std::vector<int> lens(n, 0);
  for (int w = 0; w < n; ++w) {
    const int len = std::min<int>(static_cast<int>(batch[w].size()), enc.max_len);
    lens[w] = len;
    for (int i = 0; i < len; ++i) ids[static_cast<size_t>(w) * enc.max_len + i] = batch[w][i];
  }
  enc.encode_batch(ids.data(), lens.data(), n, out.mutable_data());
  return out;
}

double capacity_fhrr(int D, int M, int trials, uint64_t seed) {
  double tot = 0.0;
  for (int t = 0; t < trials; ++t) {
    rng_t rng(seed ^ (static_cast<uint64_t>(t) << 32));
    std::vector<FHRR> roles, fillers;
    for (int i = 0; i < M; ++i) {
      roles.push_back(FHRR::random(D, rng));
      fillers.push_back(FHRR::random(D, rng));
    }
    FHRR mem(D);
    for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
    mem.normalize();
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
    tot += static_cast<double>(ok) / M;
  }
  return tot / trials;
}

double capacity_ghrr(int D, int m, double diag, int M, int trials, uint64_t seed) {
  double tot = 0.0;
  for (int t = 0; t < trials; ++t) {
    rng_t rng(seed ^ (static_cast<uint64_t>(t) << 32));
    std::vector<GHRR> roles, fillers;
    for (int i = 0; i < M; ++i) {
      roles.push_back(GHRR::random_unitary(D, m, diag, rng));
      fillers.push_back(GHRR::random_unitary(D, m, diag, rng));
    }
    GHRR mem(D, m);
    for (int i = 0; i < M; ++i) mem.add_scaled(roles[i].bind(fillers[i]), 1.0);
    mem.normalize();
    int ok = 0;
    for (int i = 0; i < M; ++i) {
      GHRR est = mem.left_unbind(roles[i]);
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
    tot += static_cast<double>(ok) / M;
  }
  return tot / trials;
}

}  // namespace

PYBIND11_MODULE(vsacore_py, m) {
  m.doc() = "VSA kernels: FHRR, GHRR, HeRR (C++ core)";

  py::class_<FHRR>(m, "FHRR")
      .def(py::init<int>(), py::arg("D"))
      .def_static("random", [](int D, uint64_t seed) {
        rng_t rng(seed);
        return FHRR::random(D, rng);
      })
      .def("bind", &FHRR::bind)
      .def("unbind", &FHRR::unbind)
      .def("sim", &FHRR::sim)
      .def("normalize", &FHRR::normalize)
      .def_property_readonly("array", &fhrr_array)
      .def_static("from_array", &fhrr_from_array)
      .def_property_readonly("D", [](const FHRR& v) { return v.D; });

  py::class_<GHRR>(m, "GHRR")
      .def(py::init<int, int>(), py::arg("D"), py::arg("m"))
      .def_static("random_unitary",
                  [](int D, int m, double diag, uint64_t seed) {
                    rng_t rng(seed);
                    return GHRR::random_unitary(D, m, diag, rng);
                  })
      .def_static("identity", &GHRR::identity)
      .def("bind", &GHRR::bind)
      .def("unbind", &GHRR::unbind)
      .def("left_unbind", &GHRR::left_unbind)
      .def("sim", &GHRR::sim)
      .def("normalize", &GHRR::normalize)
      .def_property_readonly("array", &ghrr_array)
      .def_property_readonly("D", [](const GHRR& v) { return v.D; })
      .def_property_readonly("m", [](const GHRR& v) { return v.m; });

  py::class_<HeRR>(m, "HeRR")
      .def(py::init<>())
      .def_static("monomial", &HeRR::monomial)
      .def_static("sparse_phase", [](int n, int k, int start, uint64_t seed) {
        rng_t rng(seed);
        return HeRR::sparse_phase(n, k, start, rng);
      })
      .def_static("dense_phase", [](int n, int start, uint64_t seed) {
        rng_t rng(seed);
        return HeRR::dense_phase(n, start, rng);
      })
      .def("bind", &HeRR::bind)
      .def("shift_scale", &HeRR::shift_scale, py::arg("shift"), py::arg("coeff") = cd(1, 0))
      .def("add_scaled", &HeRR::add_scaled)
      .def("inner", &HeRR::inner)
      .def("sim", &HeRR::sim)
      .def("normalize", &HeRR::normalize)
      .def_property_readonly("lo", [](const HeRR& v) { return v.lo; })
      .def_property_readonly("support", &HeRR::support)
      .def_property_readonly("array", &herr_array)
      .def_static("decode",
                  [](const HeRR& memory, const std::vector<HeRR>& roles,
                     const std::vector<HeRR>& fillers, bool peel) {
                    return HeRR::decode(memory, roles, fillers, nullptr, peel);
                  },
                  py::arg("memory"), py::arg("roles"), py::arg("fillers"),
                  py::arg("peel") = true);

  py::class_<CharEncoder>(m, "CharEncoder")
      .def(py::init<int, int, int, uint64_t>(), py::arg("vocab"), py::arg("D"),
           py::arg("max_len"), py::arg("seed"))
      .def("encode", &char_encode_array, py::arg("ids"))
      .def("encode_batch", &char_encode_batch_array, py::arg("batch"))
      .def_property_readonly("vocab", [](const CharEncoder& e) { return e.vocab; })
      .def_property_readonly("D", [](const CharEncoder& e) { return e.D; })
      .def_property_readonly("max_len", [](const CharEncoder& e) { return e.max_len; });

  m.def("capacity_fhrr", &capacity_fhrr, py::arg("D"), py::arg("M"), py::arg("trials") = 5,
        py::arg("seed") = 12345);
  m.def("capacity_ghrr", &capacity_ghrr, py::arg("D"), py::arg("m"), py::arg("diag"), py::arg("M"),
        py::arg("trials") = 5, py::arg("seed") = 12345);
}
