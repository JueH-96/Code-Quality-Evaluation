# YOUR CODE HERE
import sys

def is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    # Calculate the squares of the lengths of the sides
    AB = (x_B - x_A)**2 + (y_B - y_A)**2
    BC = (x_C - x_B)**2 + (y_C - y_B)**2
    CA = (x_A - x_C)**2 + (y_A - y_C)**2

    # Check if the square of the longest side is equal to the sum of the squares of the other two sides
    if max(AB, BC, CA) == AB:
        return AB == BC + CA
    elif max(AB, BC, CA) == BC:
        return BC == AB + CA
    else:
        return CA == AB + BC

# Read the inputs
x_A, y_A = map(int, sys.stdin.readline().split())
x_B, y_B = map(int, sys.stdin.readline().split())
x_C, y_C = map(int, sys.stdin.readline().split())

# Check if the triangle is a right triangle
if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    print("Yes")
else:
    print("No")