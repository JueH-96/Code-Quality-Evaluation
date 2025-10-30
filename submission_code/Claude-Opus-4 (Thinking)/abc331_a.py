# YOUR CODE HERE
# Read input
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Calculate next day
if d < D:
    # Not the last day of month, just increment day
    next_y, next_m, next_d = y, m, d + 1
elif m < M:
    # Last day of month but not last month, go to next month
    next_y, next_m, next_d = y, m + 1, 1
else:
    # Last day of last month, go to next year
    next_y, next_m, next_d = y + 1, 1, 1

# Print output
print(next_y, next_m, next_d)