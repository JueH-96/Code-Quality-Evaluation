import sys

def is_right_triangle(x1, y1, x2, y2, x3, y3):
    # Calculate the lengths of the sides
    a = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    b = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
    c = ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5

    # Check if the triangle is right-angled
    if a**2 + b**2 == c**2 or a**2 + c**2 == b**2 or b**2 + c**2 == a**2:
        return "Yes"
    else:
        return "No"

# Read the inputs
for line in sys.stdin:
    x1, y1, x2, y2, x3, y3 = map(int, line.split())
    print(is_right_triangle(x1, y1, x2, y2, x3, y3))