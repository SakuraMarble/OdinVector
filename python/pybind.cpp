#include "pyindex.h"

PYBIND11_MODULE(pipeannpy, m) {
  m.doc() = "PipeANN";
  m.attr("__version__") = "dev";

  py::enum_<pipeann::Metric>(m, "Metric")
      .value("L2", pipeann::Metric::L2)
      .value("COSINE", pipeann::Metric::COSINE)
      .export_values();
}