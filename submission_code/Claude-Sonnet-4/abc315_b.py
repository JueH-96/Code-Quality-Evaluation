# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))

# Calculate total days
total_days = sum(D)

# Find the middle day (1-indexed)
middle_day = (total_days + 1) // 2

# Find which month and day
cumulative_days = 0
for month in range(M):
    if cumulative_days + D[month] >= middle_day:
        # The middle day is in this month
        day_in_month = middle_day - cumulative_days
        print(month + 1, day_in_month)
        break
    cumulative_days += D[month]