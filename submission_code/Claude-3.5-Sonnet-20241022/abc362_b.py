def is_right_triangle(x1, y1, x2, y2, x3, y3):
    # Calculate squared lengths of the three sides
    side1_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    side2_sq = (x3 - x2) ** 2 + (y3 - y2) ** 2
    side3_sq = (x1 - x3) ** 2 + (y1 - y3) ** 2
    
    # Sort the squared lengths
    sides = sorted([side1_sq, side2_sq, side3_sq])
    
    # Check if Pythagorean theorem holds
    return sides[0] + sides[1] == sides[2]

# Read input
x_a, y_a = map(int, input().split())
x_b, y_b = map(int, input().split())
x_c, y_c = map(int, input().split())

# Check if it's a right triangle and print result
if is_right_triangle(x_a, y_a, x_b, y_b, x_c, y_c):
    print("Yes")
else:
    print("No")