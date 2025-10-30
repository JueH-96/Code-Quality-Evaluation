# Read the input values
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Increment the day
d += 1

# Check if the day exceeds the number of days in the month
if d > D:
    d = 1  # Reset the day to 1
    m += 1  # Increment the month

    # Check if the month exceeds the number of months in the year
    if m > M:
        m = 1  # Reset the month to 1
        y += 1  # Increment the year

# Output the result
print(y, m, d)