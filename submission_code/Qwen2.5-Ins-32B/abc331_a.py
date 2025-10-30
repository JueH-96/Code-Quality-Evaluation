import sys

# Reading input from stdin
M, D = map(int, input().split())
y, m, d = map(int, input().split())

# Incrementing the day
d += 1

# Checking if the day exceeds the month's days
if d > D:
    d = 1
    m += 1

# Checking if the month exceeds the year's months
if m > M:
    m = 1
    y += 1

# Writing the output to stdout
print(y, m, d)