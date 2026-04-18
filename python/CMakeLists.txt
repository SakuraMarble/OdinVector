# pip install "pybind11[global]"

find_package(pybind11 REQUIRED)
pybind11_add_module(pipeannpy pybind.cpp)
