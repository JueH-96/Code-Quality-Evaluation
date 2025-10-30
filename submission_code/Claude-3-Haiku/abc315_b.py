# Read the input
M = int(input())
D = list(map(int, input().split()))

# Calculate the total number of days in the year
total_days = sum(D)

# Find the middle day
middle_day = (total_days + 1) // 2

# Find the month and day of the middle day
month = 1
day = 0
while middle_day > D[month - 1]:
    middle_day -= D[month - 1]
    month += 1
day = middle_day

# Print the answer
print(month, day)