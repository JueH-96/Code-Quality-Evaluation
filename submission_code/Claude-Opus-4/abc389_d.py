# YOUR CODE HERE
R = int(input())
R_squared = R * R
count = 0

# Count squares in the first quadrant (excluding axes)
for i in range(1, R + 1):
    for j in range(1, R + 1):
        if (i + 0.5) ** 2 + (j + 0.5) ** 2 <= R_squared:
            count += 1

# Multiply by 4 for all four quadrants
count *= 4

# Count squares on the positive x-axis (excluding origin)
for i in range(1, R + 1):
    if (i + 0.5) ** 2 + 0.5 ** 2 <= R_squared:
        count += 2  # One on positive x-axis, one on negative

# Count squares on the positive y-axis (excluding origin)
for j in range(1, R + 1):
    if 0.5 ** 2 + (j + 0.5) ** 2 <= R_squared:
        count += 2  # One on positive y-axis, one on negative

# Check the square at the origin
if 0.5 ** 2 + 0.5 ** 2 <= R_squared:
    count += 1

print(count)