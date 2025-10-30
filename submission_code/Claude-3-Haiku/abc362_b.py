import math

# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate the lengths of the sides
a = math.sqrt((x_B - x_A)**2 + (y_B - y_A)**2)
b = math.sqrt((x_C - x_B)**2 + (y_C - y_B)**2)
c = math.sqrt((x_A - x_C)**2 + (y_A - y_C)**2)

# Check if the triangle is a right triangle
if round(a**2 + b**2, 6) == round(c**2, 6) or \
   round(a**2 + c**2, 6) == round(b**2, 6) or \
   round(b**2 + c**2, 6) == round(a**2, 6):
    print("Yes")
else:
    print("No")