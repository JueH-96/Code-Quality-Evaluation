# Read coordinates of the three points
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate squared distances between each pair of points
dab_sq = (x_A - x_B) ** 2 + (y_A - y_B) ** 2
dac_sq = (x_A - x_C) ** 2 + (y_A - y_C) ** 2
dbc_sq = (x_B - x_C) ** 2 + (y_B - y_C) ** 2

# Check if any combination satisfies Pythagorean theorem
if (dab_sq + dac_sq == dbc_sq) or (dab_sq + dbc_sq == dac_sq) or (dac_sq + dbc_sq == dab_sq):
    print("Yes")
else:
    print("No")