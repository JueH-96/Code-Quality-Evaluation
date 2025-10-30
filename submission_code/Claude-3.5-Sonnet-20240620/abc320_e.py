# YOUR CODE HERE
from collections import deque

# Read input
N, M = map(int, input().split())
events = [tuple(map(int, input().split())) for _ in range(M)]

# Initialize variables
noodles = [0] * N  # Amount of noodles for each person
queue = deque(range(N))  # Queue of people in line
return_times = [0] * N  # Time when each person returns to the line

for T, W, S in events:
    # Process returns
    while queue and return_times[queue[0]] <= T:
        queue.append(queue.popleft())

    # Process noodle distribution
    if queue:
        person = queue.popleft()
        noodles[person] += W
        return_times[person] = T + S
        queue.append(person)

# Print results
for amount in noodles:
    print(amount)