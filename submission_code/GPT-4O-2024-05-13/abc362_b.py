# YOUR CODE HERE
import sys
import math

input = sys.stdin.read
data = input().split()

x_A, y_A = int(data[0]), int(data[1])
x_B, y_B = int(data[2]), int(data[3])
x_C, y_C = int(data[4]), int(data[5])

# Function to calculate the squared distance between two points
def squared_distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

# Calculate the squared lengths of the sides of the triangle
AB2 = squared_distance(x_A, y_A, x_B, y_B)
BC2 = squared_distance(x_B, y_B, x_C, y_C)
CA2 = squared_distance(x_C, y_C, x_A, y_A)

# Check if any combination of sides satisfies the Pythagorean theorem
if (AB2 + BC2 == CA2) or (AB2 + CA2 == BC2) or (BC2 + CA2 == AB2):
    print("Yes")
else:
    print("No")