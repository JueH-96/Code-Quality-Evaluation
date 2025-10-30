import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

events = []
for i in range(M):
    T = int(data[3 * i + 2])
    W = int(data[3 * i + 3])
    S = int(data[3 * i + 4])
    events.append((T, W, S))

noodles = [0] * N
queue = deque(range(1, N + 1))
return_times = []

for event in events:
    T, W, S = event

    # Process returns
    while return_times and return_times[0][0] <= T:
        return_time, person = return_times.pop(0)
        queue.append(person)

    if queue:
        person = queue.popleft()
        noodles[person - 1] += W
        return_times.append((T + S, person))

    # Sort return_times to maintain order
    return_times.sort()

for amount in noodles:
    print(amount)