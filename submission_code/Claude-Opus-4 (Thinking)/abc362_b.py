# YOUR CODE HERE
# Read inputs
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate squared distances
AB_squared = (x_B - x_A)**2 + (y_B - y_A)**2
BC_squared = (x_C - x_B)**2 + (y_C - y_B)**2
CA_squared = (x_A - x_C)**2 + (y_A - y_C)**2

# Check if it's a right triangle using Pythagorean theorem
if AB_squared + BC_squared == CA_squared or \
   BC_squared + CA_squared == AB_squared or \
   CA_squared + AB_squared == BC_squared:
    print("Yes")
else:
    print("No")