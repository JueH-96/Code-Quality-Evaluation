# YOUR CODE HERE
import sys

# Read N and D
N, D = map(int, sys.stdin.readline().split())

# Read N schedules
schedules = []
for _ in range(N):
    schedules.append(sys.stdin.readline().strip())

# Determine if all people are free on each day
# all_free_on_day[j] will be True if everyone is free on day j (0-indexed), False otherwise
all_free_on_day = [False] * D # Initialize with False

for j in range(D): # Iterate through days (0 to D-1)
    # Check if day j is free for ALL people
    is_day_j_free_for_all = True
    for i in range(N): # Iterate through people (0 to N-1)
        if schedules[i][j] == 'x':
            is_day_j_free_for_all = False
            # If even one person is occupied, the day is not free for everyone
            # No need to check other people for this day
            break
    all_free_on_day[j] = is_day_j_free_for_all


# Find the maximum number of consecutive days where everyone is free
max_consecutive_days = 0
current_consecutive_days = 0

for is_free in all_free_on_day:
    if is_free:
        current_consecutive_days += 1
    else:
        # The sequence of consecutive free days is broken
        # Update the maximum if the current sequence is longer
        max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
        # Reset the current consecutive count
        current_consecutive_days = 0

# After iterating through all days, there might be a sequence ending on the last day.
# Update the maximum one last time with the final current count.
max_consecutive_days = max(max_consecutive_days, current_consecutive_days)

# Print the result
print(max_consecutive_days)