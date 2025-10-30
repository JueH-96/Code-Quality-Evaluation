# YOUR CODE HERE
import heapq
from collections import defaultdict

N, M = map(int, input().split())

# Store trains grouped by destination station
trains_to = defaultdict(list)

for _ in range(M):
    l, d, k, c, A, B = map(int, input().split())
    # Generate all trains for this schedule
    for i in range(k):
        departure_time = l + i * d
        arrival_time = departure_time + c
        trains_to[B].append((A, departure_time, arrival_time))

# Find latest departure time from each station to reach station N
# latest[s] = latest time we can depart from station s and reach N
latest = [-float('inf')] * (N + 1)
latest[N] = float('inf')  # We're already at N

# Priority queue: (-latest_departure_time, station)
pq = [(-float('inf'), N)]

while pq:
    neg_time, station = heapq.heappop(pq)
    current_latest = -neg_time
    
    if current_latest < latest[station]:
        continue
    
    # Check all trains arriving at this station
    for from_station, dep_time, arr_time in trains_to[station]:
        # If we can take this train and still reach N
        if arr_time <= latest[station]:
            # The latest we can depart from from_station is dep_time
            if dep_time > latest[from_station]:
                latest[from_station] = dep_time
                heapq.heappush(pq, (-dep_time, from_station))

# Output results
for i in range(1, N):
    if latest[i] == -float('inf'):
        print("Unreachable")
    else:
        print(latest[i])