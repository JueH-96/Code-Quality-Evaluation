import sys
from heapq import heappop, heappush

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1
Y = int(data[index])
index += 1

P = [0] * (N - 1)
T = [0] * (N - 1)
for i in range(N - 1):
    P[i] = int(data[index])
    index += 1
    T[i] = int(data[index])
    index += 1

Q = int(data[index])
index += 1
q = [0] * Q
for i in range(Q):
    q[i] = int(data[index])
    index += 1

def earliest_arrival_time(start_time):
    # Priority queue to store (current_time, current_stop)
    pq = [(start_time + X, 0)]
    visited = [False] * (N + 1)

    while pq:
        current_time, current_stop = heappop(pq)

        if current_stop == N:
            return current_time + Y

        if visited[current_stop]:
            continue

        visited[current_stop] = True

        # Walk to the next stop
        if current_stop < N:
            heappush(pq, (current_time + T[current_stop], current_stop + 1))

        # Wait for the bus
        if current_stop < N - 1:
            next_bus_time = ((current_time + P[current_stop] - 1) // P[current_stop]) * P[current_stop]
            heappush(pq, (next_bus_time + T[current_stop], current_stop + 1))

    return -1

results = []
for start_time in q:
    results.append(earliest_arrival_time(start_time))

for result in results:
    print(result)