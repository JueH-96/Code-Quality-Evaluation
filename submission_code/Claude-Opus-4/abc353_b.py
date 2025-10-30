# YOUR CODE HERE
n, k = map(int, input().split())
groups = list(map(int, input().split()))

empty_seats = k
attraction_starts = 0

for group_size in groups:
    if empty_seats < group_size:
        # Start the attraction
        attraction_starts += 1
        empty_seats = k
    
    # Guide the group to the attraction
    empty_seats -= group_size

# Start the attraction one final time after all groups are processed
attraction_starts += 1

print(attraction_starts)