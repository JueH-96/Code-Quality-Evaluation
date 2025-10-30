# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))

# Calculate total days in the year
total_days = sum(D)

# Find the position of the middle day
middle_day_pos = (total_days + 1) // 2

# Find which month and day this position corresponds to
cumulative_days = 0
for i in range(M):
    # Check if the middle day is in month i+1
    if cumulative_days + D[i] >= middle_day_pos:
        month = i + 1  # months are 1-indexed
        day = middle_day_pos - cumulative_days
        print(month, day)
        break
    cumulative_days += D[i]