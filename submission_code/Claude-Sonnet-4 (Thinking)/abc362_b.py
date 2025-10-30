# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate squared distances
d_AB_sq = (x_B - x_A) ** 2 + (y_B - y_A) ** 2
d_BC_sq = (x_C - x_B) ** 2 + (y_C - y_B) ** 2
d_AC_sq = (x_C - x_A) ** 2 + (y_C - y_A) ** 2

# Check if it's a right triangle using Pythagorean theorem
if (d_AB_sq == d_BC_sq + d_AC_sq or
    d_BC_sq == d_AB_sq + d_AC_sq or
    d_AC_sq == d_AB_sq + d_BC_sq):
    print("Yes")
else:
    print("No")