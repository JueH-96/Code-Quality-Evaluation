# YOUR CODE HERE
import heapq
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
C = int(data[3])

D = []
index = 4
for i in range(N):
    D.append([int(data[index + j]) for j in range(N)])
    index += N

# Dijkstra's algorithm to find the shortest path
def dijkstra(start, N, graph):
    dist = [float('inf')] * N
    dist[start] = 0
    pq = [(0, start)]  # (distance, node)
    heapq.heapify(pq)
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v in range(N):
            if u != v:
                # Company car
                car_time = D[u][v] * A
                if dist[v] > dist[u] + car_time:
                    dist[v] = dist[u] + car_time
                    heapq.heappush(pq, (dist[v], v))
                
                # Train
                train_time = D[u][v] * B + C
                if dist[v] > dist[u] + train_time:
                    dist[v] = dist[u] + train_time
                    heapq.heappush(pq, (dist[v], v))
    
    return dist

# Find the minimum time to travel from city 1 (index 0) to city N (index N-1)
min_time = dijkstra(0, N, D)[N-1]
print(min_time)