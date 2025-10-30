# YOUR CODE HERE
import math

x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

def dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def is_right_triangle(p1, p2, p3):
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    v3 = (p3[0] - p2[0], p3[1] - p2[1])
    
    return dot_product(v1, v2) == 0 or dot_product(v1, v3) == 0 or dot_product(v2, v3) == 0

if is_right_triangle((x_A, y_A), (x_B, y_B), (x_C, y_C)):
    print("Yes")
else:
    print("No")