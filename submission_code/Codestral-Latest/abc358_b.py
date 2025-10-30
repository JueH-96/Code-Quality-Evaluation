import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
T = list(map(int, data[2:]))

# Initialize the queue and the time when the first person finishes
queue = deque()
finish_time = T[0] + A

# Process each person
for i in range(1, N):
    # If the current person arrives before the previous person finishes
    if T[i] < finish_time:
        queue.append(T[i])
    else:
        # Process all people in the queue
        while queue:
            finish_time = max(queue.popleft() + A, finish_time)
        # Update the finish time for the current person
        finish_time = max(T[i] + A, finish_time)

    # Print the finish time for the current person
    print(finish_time)

# Process any remaining people in the queue
while queue:
    finish_time = max(queue.popleft() + A, finish_time)
    print(finish_time)