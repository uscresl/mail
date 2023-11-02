cd softgym/PyFlex/bindings
rm -rf build
mkdir build
cd build
cmake -DPYBIND11_PYTHON_VERSION=3.8 .. # for Ubuntu 20.04
# cmake -DPYBIND11_PYTHON_VERSION=3.7 .. # for Ubuntu 18.04
make -j
