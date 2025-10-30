import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Graph represented as adjacency list with (departure_time, travel_time, destination) tuples
trains = [[] for _ in range(N + 1)]

index = 2
for _ in range(M):
    l = int(data[index])
    d = int(data[index + 1])
    k = int(data[index + 2])
    c = int(data[index + 3])
    A = int(data[index + 4])
    B = int(data[index + 5])
    index += 6
    
    for j in range(k):
        departure_time = l + j * d
        trains[A].append((departure_time, c, B))

# We need to find the latest time we can arrive at station N from each station
# We will use a modified Dijkstra's algorithm to propagate the latest possible arrival times

# f(S) stores the latest time we can arrive at station N starting from station S
f = [-float('inf')] * (N + 1)
f[N] = float('inf')  # We are already at N

# Priority queue to process nodes based on the latest time we can reach N
pq = []
heapq.heappush(pq, (-f[N], N))  # (negative of latest time, station)

while pq:
    current_time, current_station = heapq.heappop(pq)
    current_time = -current_time
    
    if current_time < f[current_station]:
        continue
    
    for departure_time, travel_time, next_station in trains[current_station]:
        # Calculate the latest time we can start at next_station and still reach N
        arrival_time = departure_time + travel_time
        if arrival_time <= f[next_station]:
            continue
        
        # The latest time we can be at next_station and still reach N
        latest_start_time = min(current_time, departure_time)
        if latest_start_time > f[next_station]:
            f[next_station] = latest_start_time
            heapq.heappush(pq, (-f[next_station], next_station))

# Output the results for f(1) to f(N-1)
for station in range(1, N):
    if f[station] == -float('inf'):
        print("Unreachable")
    else:
        print(f[station])