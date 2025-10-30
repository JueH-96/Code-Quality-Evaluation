N, K = map(int, input().split())
a_b = [list(map(int, input().split())) for _ in range(N)]

# Initialize a list with N elements, all set to 0 (the maximum of integers).
max_day = [0] * N

# For each medicine, calculate the last day on which pills are taken and store it in the max_day list.
for i in range(N):
    a, b = a_b[i]
    max_day[i] = a + b - 1  # Last day of taking pills

# Find the greatest value in max_day (last day when medicine is taken)
max_day = max(max_day)

# Initialize two variables for binary search, left and right
left = 1  # Initialize the left boundary of the binary search
right = max_day + 1  # Add 1 to ensure the day is inclusive when it's the last day of medicine

# Perform binary search to find the minimum day Takahashi needs to take K pills or fewer
while left < right:
    mid_day = (left + right) // 2 # Find the middle day
    daily_pills = 0  # Initialize the count of daily pills for the middle day

    for i in range(N):
        a, b = a_b[i]
        if mid_day <= a: # If mid_day is earlier or on the same day the medicine starts, continue to next medicine
            continue
        elif mid_day <= a + b: # If mid_day falls within the taking period, calculate pills taken
            pills_taken = a_b[i][1] - ( (mid_day - a - 1) * (mid_day - a) ) // 2
            daily_pills += pills_taken
        else: # If mid_day is after the medicine taking period, add minimum pills
            daily_pills += a_b[i][1]

    if daily_pills > K:  # If pills taken are more than K, search in the right half
        left = mid_day + 1
    else:  # If pills taken are K or fewer, search in the left half or the middle is the answer
        right = mid_day

print(left)