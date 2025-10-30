import math

# Read input values
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Function to calculate the square of the distance between two points
def distance_squared(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

# Calculate the squared distances between the points
d_AB_squared = distance_squared(x_A, y_A, x_B, y_B)
d_BC_squared = distance_squared(x_B, y_B, x_C, y_C)
d_CA_squared = distance_squared(x_C, y_C, x_A, y_A)

# Check if the triangle is a right triangle
if (d_AB_squared + d_BC_squared == d_CA_squared or
    d_AB_squared + d_CA_squared == d_BC_squared or
    d_BC_squared + d_CA_squared == d_AB_squared):
    print("Yes")
else:
    print("No")