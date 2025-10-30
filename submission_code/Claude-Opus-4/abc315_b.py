# YOUR CODE HERE
M = int(input())
days = list(map(int, input().split()))

# Calculate total days
total_days = sum(days)

# Find the middle day position
middle_position = (total_days + 1) // 2

# Find which month and day
cumulative_days = 0
for month in range(M):
    if cumulative_days + days[month] >= middle_position:
        # The middle day is in this month
        day_in_month = middle_position - cumulative_days
        print(month + 1, day_in_month)
        break
    cumulative_days += days[month]