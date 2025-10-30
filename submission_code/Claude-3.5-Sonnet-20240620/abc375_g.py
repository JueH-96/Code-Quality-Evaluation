# YOUR CODE HERE
import sys
from heapq import heappush, heappop
from collections import defaultdict

def dijkstra(graph, start, end, excluded_edge=None):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heappop(pq)
        
        if current_node == end:
            return current_distance
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            if (current_node, neighbor) == excluded_edge or (neighbor, current_node) == excluded_edge:
                continue
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(pq, (distance, neighbor))
    
    return float('inf')

def main():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    edges = []

    for _ in range(M):
        a, b, c = map(int, input().split())
        graph[a].append((b, c))
        graph[b].append((a, c))
        edges.append((a, b, c))

    all_roads_distance = dijkstra(graph, 1, N)

    for a, b, _ in edges:
        distance_without_road = dijkstra(graph, 1, N, excluded_edge=(a, b))
        if distance_without_road != all_roads_distance:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()