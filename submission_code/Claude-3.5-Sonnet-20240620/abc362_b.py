# YOUR CODE HERE
def distance_squared(x1, y1, x2, y2):
    return (x2 - x1)**2 + (y2 - y1)**2

def is_right_triangle(a, b, c):
    sides = sorted([a, b, c])
    return sides[0] + sides[1] == sides[2]

# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate squared distances
AB = distance_squared(x_A, y_A, x_B, y_B)
BC = distance_squared(x_B, y_B, x_C, y_C)
CA = distance_squared(x_C, y_C, x_A, y_A)

# Check if it's a right triangle
if is_right_triangle(AB, BC, CA):
    print("Yes")
else:
    print("No")