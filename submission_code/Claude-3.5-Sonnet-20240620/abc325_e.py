# YOUR CODE HERE
import sys
from heapq import heappush, heappop

def dijkstra(graph, start, end, N, A, B, C):
    distances = [[float('inf')] * 2 for _ in range(N)]
    distances[start][0] = 0  # [car_distance, train_distance]
    pq = [(0, start, 0)]  # (distance, node, mode) where mode 0 is car, 1 is train

    while pq:
        dist, node, mode = heappop(pq)
        
        if node == end:
            return int(dist)
        
        if dist > distances[node][mode]:
            continue
        
        for neighbor in range(N):
            if neighbor == node:
                continue
            
            # Car mode
            if mode == 0:
                new_dist = dist + graph[node][neighbor] * A
                if new_dist < distances[neighbor][0]:
                    distances[neighbor][0] = new_dist
                    heappush(pq, (new_dist, neighbor, 0))
                
                # Switch to train
                train_dist = dist + graph[node][neighbor] * B + C
                if train_dist < distances[neighbor][1]:
                    distances[neighbor][1] = train_dist
                    heappush(pq, (train_dist, neighbor, 1))
            
            # Train mode
            else:
                new_dist = dist + graph[node][neighbor] * B + C
                if new_dist < distances[neighbor][1]:
                    distances[neighbor][1] = new_dist
                    heappush(pq, (new_dist, neighbor, 1))

    return float('inf')

# Read input
N, A, B, C = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(N)]

# Solve and print result
print(dijkstra(graph, 0, N-1, N, A, B, C))