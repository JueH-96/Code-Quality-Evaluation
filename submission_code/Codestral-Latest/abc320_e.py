import heapq

# Read input
N, M = map(int, input().split())
events = []

for _ in range(M):
    T, W, S = map(int, input().split())
    events.append((T, W, S))

# Sort events by time
events.sort()

# Initialize variables
noodles = [0] * N
queue = list(range(1, N + 1))
heapq.heapify(queue)
return_times = []

# Process events
for T, W, S in events:
    while return_times and return_times[0][0] <= T:
        _, person = heapq.heappop(return_times)
        heapq.heappush(queue, person)

    if queue:
        person = heapq.heappop(queue)
        noodles[person - 1] += W
        heapq.heappush(return_times, (T + S, person))

# Output the result
for i in range(N):
    print(noodles[i])