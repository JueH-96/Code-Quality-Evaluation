import sys

input = sys.stdin.read
data = input().split()

M = int(data[0])
D = int(data[1])
y = int(data[2])
m = int(data[3])
d = int(data[4])

# Increment the day
d += 1

# Check if the day exceeds the number of days in a month
if d > D:
    d = 1
    m += 1

# Check if the month exceeds the number of months in a year
if m > M:
    m = 1
    y += 1

# Print the result
print(y, m, d)