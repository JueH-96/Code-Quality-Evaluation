import collections

# Read N and K from the first line of input
N, K = map(int, input().split())

# Read the group sizes A_i from the second line of input
A = list(map(int, input().split()))

# Initialize a deque (double-ended queue) for efficient popping from the left
queue = collections.deque(A)

# Initialize the number of empty seats in the attraction.
# Initially, the attraction is empty, so all K seats are available.
current_empty_seats = K

# Initialize the counter for the number of times the attraction has started.
attraction_starts = 0

# Simulate the guidance process
# The loop continues as long as there are groups in the queue
while queue:
    # Get the size of the group at the front of the queue without removing it yet.
    # We use queue[0] to peek at the front element.
    group_size = queue[0]

    # Check if the current number of empty seats is less than the group size
    if current_empty_seats < group_size:
        # If there are not enough seats, the attraction must be started.
        attraction_starts += 1
        # After starting, the attraction becomes empty again, so reset seats to K.
        current_empty_seats = K
        # The group that caused the attraction to start remains at the front
        # of the queue. It will be considered again in the next iteration,
        # now that the attraction has full capacity.
    else:
        # If there are enough seats, guide the group into the attraction.
        # Remove the group from the front of the queue.
        queue.popleft()
        # Decrease the number of empty seats by the size of the guided group.
        current_empty_seats -= group_size

# After the loop finishes, it means all groups have been guided into the attraction.
# According to the problem statement ("If there are no groups in the queue,
# start the attraction and end the guidance."), the attraction starts one final time
# with the last batch of people who have entered it.
attraction_starts += 1

# Print the total number of times the attraction was started.
print(attraction_starts)