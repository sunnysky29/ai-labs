

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    `python3 -m pybind11 --includes` \
    math_lib.cpp -o math_lib`python3-config --extension-suffix`
