import heapq
import sys

# Read input
line1 = sys.stdin.readline().split()
N = int(line1[0])
A = int(line1[1])
B = int(line1[2])
C = int(line1[3])

D = []
for _ in range(N):
    D.append(list(map(int, sys.stdin.readline().split())))

# Graph setup
# Nodes 0 to N-1: cities 1 to N by car
# Nodes N to 2N-1: cities 1 to N by train
num_nodes = 2 * N
dist = [float('inf')] * num_nodes

# Start at city 1 by car (node 0, which is 0-indexed city 0)
dist[0] = 0
pq = [(0, 0)] # (time, node)

while pq:
    current_time, u = heapq.heappop(pq)

    # If we already found a shorter path to u, skip this one
    if current_time > dist[u]:
        continue

    current_city_idx = u % N # 0-indexed city index (0 to N-1)

    if u < N: # Currently using a car (nodes 0 to N-1 representing cities 1 to N)
        # Option 1: Switch to train at current city
        # The node for the current city by train is N + current_city_idx
        train_node = N + current_city_idx
        # Switching takes 0 time
        # Relax edge from u (current city car) to train_node (current city train)
        if dist[u] < dist[train_node]: # dist[u] + 0 < dist[train_node]
            dist[train_node] = dist[u]
            heapq.heappush(pq, (dist[train_node], train_node))

        # Option 2: Travel by car to any city j
        for next_city_idx in range(N):
            # The node for next city by car is next_city_idx
            next_node = next_city_idx
            # Travel time from city current_city_idx+1 to city next_city_idx+1 by car
            travel_time = D[current_city_idx][next_city_idx] * A
            # Relax edge from u (current city car) to next_node (next city car)
            if dist[u] + travel_time < dist[next_node]:
                dist[next_node] = dist[u] + travel_time
                heapq.heappush(pq, (dist[next_node], next_node))

    else: # Currently using a train (nodes N to 2N-1 representing cities 1 to N)
        # Cannot switch back to car.
        # Option 1: Travel by train to any city j
        for next_city_idx in range(N):
            # The node for next city by train is N + next_city_idx
            next_node = N + next_city_idx
            # Travel time from city current_city_idx+1 to city next_city_idx+1 by train
            travel_time = D[current_city_idx][next_city_idx] * B + C
            # Relax edge from u (current city train) to next_node (next city train)
            if dist[u] + travel_time < dist[next_node]:
                dist[next_node] = dist[u] + travel_time
                heapq.heappush(pq, (dist[next_node], next_node))

# The destination is city N (0-indexed N-1).
# It can be reached either by car (node N-1) or by train (node 2N-1).
# The minimum time is the minimum of the distances to these two nodes.
# Node for City N by car is N - 1.
# Node for City N by train is N + (N - 1) = 2*N - 1.
ans = min(dist[N - 1], dist[2 * N - 1])

# Print the answer
print(ans)