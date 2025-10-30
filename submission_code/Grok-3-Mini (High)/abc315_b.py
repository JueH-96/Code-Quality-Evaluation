# Read input
M = int(input())
D = list(map(int, input().split()))

# Calculate total sum and middle day index
total_sum = sum(D)
middle = (total_sum + 1) // 2

# Initialize cumulative sum before current month
cum_before = 0

# Iterate through each month to find which month and day the middle day falls on
for month in range(1, M + 1):
    days = D[month - 1]
    if middle <= cum_before + days:
        b = middle - cum_before
        print(month, b)
        break  # Exit loop after finding the middle day
    else:
        cum_before += days  # Update cumulative sum if middle day not in this month