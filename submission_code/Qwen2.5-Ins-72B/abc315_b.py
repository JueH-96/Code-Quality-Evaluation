# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

M = int(data[0])
D = list(map(int, data[1:]))

# Calculate the total number of days in the year
total_days = sum(D)

# Calculate the middle day
middle_day = (total_days + 1) // 2

# Find the month and day of the middle day
current_day = 0
for month in range(M):
    current_day += D[month]
    if current_day >= middle_day:
        day = middle_day - (current_day - D[month])
        break

# Print the result
print(f"{month + 1} {day}")