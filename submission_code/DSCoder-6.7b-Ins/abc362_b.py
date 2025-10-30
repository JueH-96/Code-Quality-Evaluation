# YOUR CODE HERE

# Read the inputs
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate the distances between the points
AB = ((x_B - x_A)**2 + (y_B - y_A)**2)**0.5
BC = ((x_C - x_B)**2 + (y_C - y_B)**2)**0.5
AC = ((x_C - x_A)**2 + (y_C - y_A)**2)**0.5

# Check if the triangle is a right triangle
if AB**2 + BC**2 == AC**2 or AB**2 + AC**2 == BC**2 or BC**2 + AC**2 == AB**2:
    print("Yes")
else:
    print("No")