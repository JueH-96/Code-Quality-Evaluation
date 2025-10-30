# YOUR CODE HERE
import sys

def is_right_triangle(x1, y1, x2, y2, x3, y3):
    # Calculate the squared lengths of the sides
    a2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    b2 = (x3 - x2) ** 2 + (y3 - y2) ** 2
    c2 = (x3 - x1) ** 2 + (y3 - y1) ** 2
    
    # Check if any combination of these sides satisfies the Pythagorean theorem
    return a2 + b2 == c2 or a2 + c2 == b2 or b2 + c2 == a2

# Read input
input = sys.stdin.read().split()
x1, y1 = int(input[0]), int(input[1])
x2, y2 = int(input[2]), int(input[3])
x3, y3 = int(input[4]), int(input[5])

# Determine if the triangle is a right triangle
if is_right_triangle(x1, y1, x2, y2, x3, y3):
    print("Yes")
else:
    print("No")