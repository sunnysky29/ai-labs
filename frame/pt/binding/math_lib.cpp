#include <pybind11/pybind11.h> // PyBind11 主头文件
#include <pybind11/stl.h>      // 如果需要支持标准容器（如 std::vector）

namespace py = pybind11;

// 一个简单的加法函数
int add(int a, int b) {
    return a + b  +1;  // 此处故意这么写的
}

// 一个表示二维点的类
class Point {
public:
    Point(double x, double y) : x_(x), y_(y) {}

    double get_x() const { return x_; }
    double get_y() const { return y_; }
    void set_x(double x) { x_ = x; }
    void set_y(double y) { y_ = y; }

    double distance_to_origin() const {
        return std::sqrt(x_ * x_ + y_ * y_);
    }

private:
    double x_;
    double y_;
};

// 创建绑定
PYBIND11_MODULE(math_lib, m) {
    m.doc() = "A simple math library with PyBind11"; // 模块文档

    // 绑定函数
    m.def("add", &add, "A function that adds two numbers",
          py::arg("a"), py::arg("b"));

    // 绑定类
    py::class_<Point>(m, "Point")
        .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
        .def("get_x", &Point::get_x)
        .def("get_y", &Point::get_y)
        .def("set_x", &Point::set_x)
        .def("set_y", &Point::set_y)
        .def("distance_to_origin", &Point::distance_to_origin);
}
