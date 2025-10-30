# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Calculate vectors for the three sides
# Vector AB
AB_x = x_B - x_A
AB_y = y_B - y_A

# Vector BC
BC_x = x_C - x_B
BC_y = y_C - y_B

# Vector CA
CA_x = x_A - x_C
CA_y = y_A - y_C

# Check if any two vectors are perpendicular (dot product = 0)
# At vertex A: vectors AB and AC (which is -CA)
dot_A = AB_x * (-CA_x) + AB_y * (-CA_y)

# At vertex B: vectors BA (which is -AB) and BC
dot_B = (-AB_x) * BC_x + (-AB_y) * BC_y

# At vertex C: vectors CB (which is -BC) and CA
dot_C = (-BC_x) * CA_x + (-BC_y) * CA_y

# If any dot product is zero, we have a right angle
if dot_A == 0 or dot_B == 0 or dot_C == 0:
    print("Yes")
else:
    print("No")