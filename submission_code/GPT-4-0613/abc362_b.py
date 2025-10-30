# YOUR CODE HERE
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

AB = (x_B - x_A)**2 + (y_B - y_A)**2
BC = (x_C - x_B)**2 + (y_C - y_B)**2
AC = (x_C - x_A)**2 + (y_C - y_A)**2

if AB + BC == AC or AB + AC == BC or BC + AC == AB:
    print("Yes")
else:
    print("No")