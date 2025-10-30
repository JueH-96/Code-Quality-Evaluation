# Read the input from stdin
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Calculate the next day
d += 1

# Check if the day exceeds the number of days in the month
if d > D:
    d = 1
    m += 1

# Check if the month exceeds the number of months in the year
if m > M:
    m = 1
    y += 1

# Print the result
print(y, m, d)