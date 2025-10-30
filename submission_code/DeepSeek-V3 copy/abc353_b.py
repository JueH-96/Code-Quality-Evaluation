# Read input values
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the number of empty seats and the count of attraction starts
empty_seats = K
attraction_starts = 0

# Iterate through each group in the queue
for group in A:
    if empty_seats < group:
        # Start the attraction and reset the empty seats
        attraction_starts += 1
        empty_seats = K
    # Guide the group to the attraction
    empty_seats -= group

# After processing all groups, start the attraction one more time if there are any remaining people
if empty_seats != K:
    attraction_starts += 1

# Print the total number of attraction starts
print(attraction_starts)