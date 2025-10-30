import sys

# Read input
# N: number of groups
# K: capacity of the attraction
N, K = map(int, sys.stdin.readline().split())

# A: list of group sizes
A = list(map(int, sys.stdin.readline().split()))

# Use a list as a queue.
# Using list.pop(0) is O(N), but with N <= 100, this is efficient enough.
# For larger N, collections.deque provides O(1) popleft, which would be better.
queue = A

# Initialize state
# empty_seats: current number of available seats in the attraction
empty_seats = K
# starts: counter for the number of times the attraction is started
starts = 0

# Simulate the guidance process
# The process continues as long as there are groups waiting in the queue.
# The decision to "Go back to step 1" after each action means we re-check
# if the queue is empty at the beginning of each iteration.
while queue:
    # Get the size of the group at the front of the queue.
    # We check the first element without removing it yet.
    current_group_size = queue[0]

    # Compare the front group's size with the available seats.
    if current_group_size > empty_seats:
        # Case 1: Not enough seats for the group.
        # The attraction must be started.
        starts += 1 # Increment the start counter.
        empty_seats = K # The attraction becomes empty and available again.
        # The current group remains at the front of the queue;
        # it will be considered again in the next iteration with the reset seats.
    else:
        # Case 2: Enough seats for the group.
        # Guide the entire group into the attraction.
        queue.pop(0) # Remove the group from the front of the queue.
        empty_seats -= current_group_size # Decrease the number of empty seats.
        # The loop continues, and the next group (if any) will be considered
        # in the following iteration.

# After the loop finishes, it means the queue is empty.
# According to Step 1 of the procedure ("If there are no groups in the queue,
# start the attraction and end the guidance"), a final start occurs.
starts += 1

# Print the total number of times the attraction was started.
print(starts)