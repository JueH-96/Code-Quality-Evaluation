import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Simulate the process
groups = collections.deque(A)
empty_seats = K
start_count = 0

while True:
    if not groups:
        # No groups left, start the attraction and end
        start_count += 1
        break
    front_group_size = groups[0]  # Peek at the front group size
    if empty_seats < front_group_size:
        # Not enough seats, start the attraction
        start_count += 1
        empty_seats = K  # Reset empty seats
    else:
        # Enough seats, guide the group to the attraction
        groups.popleft()  # Remove the front group
        empty_seats -= front_group_size  # Decrease empty seats

# Output the result
print(start_count)