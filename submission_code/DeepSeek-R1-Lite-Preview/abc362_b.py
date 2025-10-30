# Read input coordinates
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate squared distances
ab_sq = (x_B - x_A)**2 + (y_B - y_A)**2
ac_sq = (x_C - x_A)**2 + (y_C - y_A)**2
bc_sq = (x_C - x_B)**2 + (y_C - y_B)**2

# Check if any combination satisfies the Pythagorean theorem
if (ab_sq + ac_sq == bc_sq) or (ab_sq + bc_sq == ac_sq) or (ac_sq + bc_sq == ab_sq):
    print("Yes")
else:
    print("No")