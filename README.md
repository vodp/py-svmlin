# py-svmlin
This is a Python wrapper of the C++ library [SVMLIN](http://vikas.sindhwani.org/svmlin.html). I have tested the wrapper version and confirm mostly identical results performed to the original source code. However, the code may contain bugs. Your contributions are welcome. In order to make SVMLIN compatible with Python CTypes, I modified a bit the C++ code, mostly in function interfaces.

### HOWTO
1. Just type `make` and the library `libtsvm.so` is compiled. 
2. In python shell, `import svmlin`. If you see `libtsvm library loaded.` then you are done.

### Dependencies
1. Numpy
2. Scipy
3. Sklearn

### Usage

