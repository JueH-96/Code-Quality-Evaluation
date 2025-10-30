import sys

def is_right_triangle(x1, y1, x2, y2, x3, y3):
    # Calculate squared lengths of the sides
    a2 = (x2 - x1)**2 + (y2 - y1)**2
    b2 = (x3 - x1)**2 + (y3 - y1)**2
    c2 = (x3 - x2)**2 + (y3 - y2)**2
    
    # Check if any of the angles is a right angle using Pythagorean theorem
    if a2 + b2 == c2 or a2 + c2 == b2 or b2 + c2 == a2:
        return "Yes"
    else:
        return "No"

# Read inputs
x1, y1 = map(int, input().split())
x2, y2 = map(int, input().split())
x3, y3 = map(int, input().split())

# Determine if the triangle is a right triangle
result = is_right_triangle(x1, y1, x2, y2, x3, y3)
print(result)