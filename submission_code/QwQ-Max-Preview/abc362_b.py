# Read the coordinates of the three points
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate vectors for each possible right angle scenario
# Check angle at A: vectors AB and AC
ab_x = x_B - x_A
ab_y = y_B - y_A
ac_x = x_C - x_A
ac_y = y_C - y_A
dot_A = ab_x * ac_x + ab_y * ac_y

# Check angle at B: vectors BA and BC
ba_x = x_A - x_B
ba_y = y_A - y_B
bc_x = x_C - x_B
bc_y = y_C - y_B
dot_B = ba_x * bc_x + ba_y * bc_y

# Check angle at C: vectors CA and CB
ca_x = x_A - x_C
ca_y = y_A - y_C
cb_x = x_B - x_C
cb_y = y_B - y_C
dot_C = ca_x * cb_x + ca_y * cb_y

# Determine if any of the dot products are zero (indicating perpendicular vectors)
if dot_A == 0 or dot_B == 0 or dot_C == 0:
    print("Yes")
else:
    print("No")