import sys
import heapq

input = sys.stdin.read
data = input().split()

index = 0

N = int(data[index])
M = int(data[index + 1])
index += 2

adj = [[] for _ in range(N + 1)]
bridges = []

for i in range(M):
    U = int(data[index])
    V = int(data[index + 1])
    T = int(data[index + 2])
    index += 3
    adj[U].append((V, T, i + 1))
    adj[V].append((U, T, i + 1))
    bridges.append((U, V, T))

Q = int(data[index])
index += 1

queries = []
for _ in range(Q):
    K = int(data[index])
    index += 1
    B = list(map(int, data[index:index + K]))
    index += K
    queries.append(B)

def dijkstra(source, target, must_use_bridges):
    # Create a dictionary to map bridge indices to their respective weights and nodes
    bridge_dict = {}
    for b in must_use_bridges:
        u, v, t = bridges[b - 1]
        bridge_dict[b] = (u, v, t)
    
    # Priority queue for Dijkstra's algorithm
    pq = []
    # Distance to each node, start with infinity
    dist = [float('inf')] * (N + 1)
    dist[source] = 0
    # Start from the source
    heapq.heappush(pq, (0, source))
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight, bridge_index in adj[u]:
            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    
    # Calculate the minimum distance using all must-use bridges
    min_dist = dist[target]
    
    # Try all permutations of the must-use bridges to find the minimum path that uses all of them
    from itertools import permutations
    for perm in permutations(must_use_bridges):
        # Start from the source
        current_position = source
        current_cost = 0
        valid = True
        
        for bridge_index in perm:
            u, v, t = bridge_dict[bridge_index]
            
            # Calculate the shortest path to the start of this bridge
            if dist[u] < dist[v]:
                # Prefer to start from u
                bridge_start = u
                bridge_end = v
            else:
                # Prefer to start from v
                bridge_start = v
                bridge_end = u
            
            # Add the cost to reach the start of the bridge
            current_cost += dist[bridge_start] + t
            # Now simulate that we are at the end of the bridge
            current_position = bridge_end
            # Update distances from the current position
            dist = [float('inf')] * (N + 1)
            dist[current_position] = 0
            pq = [(0, current_position)]
            while pq:
                d, node = heapq.heappop(pq)
                if d > dist[node]:
                    continue
                for neighbor, weight, _ in adj[node]:
                    if dist[neighbor] > dist[node] + weight:
                        dist[neighbor] = dist[node] + weight
                        heapq.heappush(pq, (dist[neighbor], neighbor))
        
        # Finally, add the cost from the last bridge end to the target
        current_cost += dist[target]
        min_dist = min(min_dist, current_cost)
    
    return min_dist

results = []
for query in queries:
    result = dijkstra(1, N, query)
    results.append(result)

for result in results:
    print(result)