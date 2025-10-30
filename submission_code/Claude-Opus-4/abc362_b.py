# YOUR CODE HERE
# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Check angle at A
# Vectors AB and AC
AB_x = x_B - x_A
AB_y = y_B - y_A
AC_x = x_C - x_A
AC_y = y_C - y_A
dot_A = AB_x * AC_x + AB_y * AC_y

# Check angle at B
# Vectors BA and BC
BA_x = x_A - x_B
BA_y = y_A - y_B
BC_x = x_C - x_B
BC_y = y_C - y_B
dot_B = BA_x * BC_x + BA_y * BC_y

# Check angle at C
# Vectors CA and CB
CA_x = x_A - x_C
CA_y = y_A - y_C
CB_x = x_B - x_C
CB_y = y_B - y_C
dot_C = CA_x * CB_x + CA_y * CB_y

# Check if any angle is 90 degrees (dot product is 0)
if dot_A == 0 or dot_B == 0 or dot_C == 0:
    print("Yes")
else:
    print("No")