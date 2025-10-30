# YOUR CODE HERE
import sys
x_A, y_A = map(int, sys.stdin.readline().split())
x_B, y_B = map(int, sys.stdin.readline().split())
x_C, y_C = map(int, sys.stdin.readline().split())

def squared_distance(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

AB2 = squared_distance(x_A, y_A, x_B, y_B)
AC2 = squared_distance(x_A, y_A, x_C, y_C)
BC2 = squared_distance(x_B, y_B, x_C, y_C)

sides = sorted([AB2, AC2, BC2])

if sides[0] + sides[1] == sides[2]:
    print("Yes")
else:
    print("No")