# YOUR CODE HERE
# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Initialize
empty_seats = K
attraction_starts = 0
queue_index = 0

# Process the queue
while queue_index < N:
    front_group_size = A[queue_index]
    
    if empty_seats < front_group_size:
        # Start the attraction
        attraction_starts += 1
        empty_seats = K
    else:
        # Guide the group
        empty_seats -= front_group_size
        queue_index += 1

# Start the attraction one last time when queue is empty
attraction_starts += 1

print(attraction_starts)