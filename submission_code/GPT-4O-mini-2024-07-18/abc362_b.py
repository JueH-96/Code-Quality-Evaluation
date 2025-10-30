def is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    # Calculate the squared lengths of the sides of the triangle
    AB_squared = (x_B - x_A) ** 2 + (y_B - y_A) ** 2
    BC_squared = (x_C - x_B) ** 2 + (y_C - y_B) ** 2
    CA_squared = (x_A - x_C) ** 2 + (y_A - y_C) ** 2
    
    # Check for the Pythagorean theorem
    return (AB_squared + BC_squared == CA_squared or
            AB_squared + CA_squared == BC_squared or
            BC_squared + CA_squared == AB_squared)

import sys

# Read input from stdin
input_data = sys.stdin.read().strip().splitlines()
x_A, y_A = map(int, input_data[0].split())
x_B, y_B = map(int, input_data[1].split())
x_C, y_C = map(int, input_data[2].split())

# Determine if the triangle is a right triangle
if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    print("Yes")
else:
    print("No")