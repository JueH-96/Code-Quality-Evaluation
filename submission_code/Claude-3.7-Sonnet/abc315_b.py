# Read input
M = int(input())
D = list(map(int, input().split()))

# Calculate total days in the year
total_days = sum(D)

# Find the middle day index
middle_day = (total_days + 1) // 2  # Integer division since the result will be an integer (total days is odd)

# Locate which month and day corresponds to the middle day
day_counter = 0
for month in range(1, M + 1):
    if day_counter + D[month - 1] >= middle_day:
        day = middle_day - day_counter
        print(month, day)
        break
    day_counter += D[month - 1]