# YOUR CODE HERE
n, k = map(int, input().split())
groups = list(map(int, input().split()))

empty_seats = k
start_count = 0

for group_size in groups:
    # If current group can't fit, start the attraction
    if group_size > empty_seats:
        start_count += 1
        empty_seats = k
    
    # Guide the group to the attraction
    empty_seats -= group_size

# If there are people seated at the end, start the attraction one final time
if empty_seats < k:
    start_count += 1

print(start_count)