# YOUR CODE HERE
import math

def is_right_triangle(x1, y1, x2, y2, x3, y3):
    # Calculate the squared lengths of the sides of the triangle
    a2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    b2 = (x2 - x3) ** 2 + (y2 - y3) ** 2
    c2 = (x3 - x1) ** 2 + (y3 - y1) ** 2
    
    # Sort the squared lengths to check the Pythagorean theorem
    sides = sorted([a2, b2, c2])
    
    # Check if the sum of the squares of the two smaller sides equals the square of the largest side
    return sides[0] + sides[1] == sides[2]

# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Determine if the triangle is a right triangle
if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    print("Yes")
else:
    print("No")