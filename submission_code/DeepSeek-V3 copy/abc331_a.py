# Read the input values
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Increment the day
d += 1

# Check if the day exceeds the month's days
if d > D:
    d = 1
    m += 1
    # Check if the month exceeds the year's months
    if m > M:
        m = 1
        y += 1

# Print the result
print(y, m, d)