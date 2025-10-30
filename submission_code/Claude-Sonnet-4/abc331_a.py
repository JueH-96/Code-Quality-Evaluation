# YOUR CODE HERE
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Check if we need to increment day, month, or year
if d < D:
    # Just increment the day
    d += 1
else:
    # d == D, so we're at the last day of the month
    d = 1  # Reset to first day
    if m < M:
        # Not the last month, so increment month
        m += 1
    else:
        # m == M, so we're at the last month of the year
        m = 1  # Reset to first month
        y += 1  # Increment year

print(y, m, d)