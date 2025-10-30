# Read the three points from input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate squared distances
AB_sq = (x_B - x_A)**2 + (y_B - y_A)**2
BC_sq = (x_C - x_B)**2 + (y_C - y_B)**2
CA_sq = (x_A - x_C)**2 + (y_A - y_C)**2

# Check if any combination satisfies Pythagorean theorem
if (AB_sq == BC_sq + CA_sq) or (BC_sq == AB_sq + CA_sq) or (CA_sq == AB_sq + BC_sq):
    print("Yes")
else:
    print("No")