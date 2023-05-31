# a = 12
# print(type(a))
#
# a = (a, a)
# print(type(a))
# print(a[0])
# print(a[1])
# import os
#
# a = open("../data_set/a.txt")
# str = a.read(20)
# print(str)

# print(os.getcwd())

def f1(x1, x2, x3):
    return x1 - 4 * x2 + x3


def f2(x1, x2, x3):
    return x1 + 2 * x2 + 2 * x3


def f3(x1, x2, x3):
    return x1 ** 2 + x2 ** 2 + x3 ** 2

#
# -2.0002
# 1.00007596
# print(f1(-0.5194, 0.1074, -0.8478))
# print(f2(-0.5194, 0.1074, -0.8478))
# print(f3(-0.5194, 0.1074, -0.8478))
print(f1(-0.247080, 0.919917,  -0.258480))
print(f2(-0.233758, 0.935033,  -0.233758))
print(f3(-0.247080, 0.919917,  -0.258480))

