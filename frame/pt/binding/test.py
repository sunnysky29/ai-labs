import math_lib

# 测试函数
result = math_lib.add(10, 20)
print(f"10 + 20 = {result}")

# 测试类
point = math_lib.Point(3.0, 4.0)
print(f"Point({point.get_x()}, {point.get_y()})")
print(f"Distance to origin: {point.distance_to_origin()}")

# 修改属性
point.set_x(6.0)
point.set_y(8.0)
print(f"Point({point.get_x()}, {point.get_y()})")
print(f"Distance to origin: {point.distance_to_origin()}")
