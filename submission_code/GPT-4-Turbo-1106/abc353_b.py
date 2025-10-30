# Read N and K from input
N, K = map(int, input().split())

# Read the list of groups from input
A = list(map(int, input().split()))

# Initialize the number of times the attraction is started
attraction_starts = 0

# Initialize the number of empty seats
empty_seats = K

# Iterate over the groups
for group in A:
    # If the number of empty seats is less than the number of people in the group
    if empty_seats < group:
        # Start the attraction
        attraction_starts += 1
        # Reset the number of empty seats
        empty_seats = K
    # Guide the group to the attraction
    empty_seats -= group

# Start the attraction for the last time if there are no more groups
if empty_seats < K:
    attraction_starts += 1

# Print the number of times the attraction is started
print(attraction_starts)