# YOUR CODE HERE
import sys
import math
from heapq import heappop, heappush

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def dijkstra_min_path_with_penalty(N, checkpoints):
    # Graph adjacency list
    graph = [[] for _ in range(N)]
    
    # Fill the graph with distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = euclidean_distance(checkpoints[i][0], checkpoints[i][1], checkpoints[j][0], checkpoints[j][1])
            graph[i].append((j, dist))
            graph[j].append((i, dist))
    
    # Priority queue for Dijkstra's algorithm
    pq = [(0, 0, 0)]  # (current_cost, current_node, skipped_checkpoints)
    visited = {}
    
    while pq:
        current_cost, current_node, skipped_checkpoints = heappop(pq)
        
        if (current_node, skipped_checkpoints) in visited:
            continue
        
        visited[(current_node, skipped_checkpoints)] = current_cost
        
        if current_node == N - 1:
            penalty = 2 ** (skipped_checkpoints - 1) if skipped_checkpoints > 0 else 0
            return current_cost + penalty
        
        for neighbor, distance in graph[current_node]:
            if neighbor == current_node + 1:
                heappush(pq, (current_cost + distance, neighbor, skipped_checkpoints))
            elif neighbor > current_node + 1:
                heappush(pq, (current_cost + distance, neighbor, skipped_checkpoints + (neighbor - current_node - 1)))
    
    return float('inf')

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    checkpoints = []
    
    for i in range(N):
        x = int(data[2 * i + 1])
        y = int(data[2 * i + 2])
        checkpoints.append((x, y))
    
    result = dijkstra_min_path_with_penalty(N, checkpoints)
    print(f"{result:.20f}")

if __name__ == "__main__":
    main()