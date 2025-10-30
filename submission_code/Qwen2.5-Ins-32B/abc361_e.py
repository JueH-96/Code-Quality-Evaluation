import sys
from heapq import heappop, heappush

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        current_dist, current_node = heappop(heap)
        if current_dist > dist[current_node]:
            continue
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heappush(heap, (distance, neighbor))
    return dist

def main():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    idx = 1
    graph = {i: [] for i in range(1, n + 1)}
    
    for _ in range(n - 1):
        a = int(data[idx])
        b = int(data[idx + 1])
        c = int(data[idx + 2])
        graph[a].append((b, c))
        graph[b].append((a, c))
        idx += 3
    
    # Find the longest path from any node
    start = 1
    dist1 = dijkstra(graph, start)
    farthest_node = max(dist1, key=dist1.get)
    dist2 = dijkstra(graph, farthest_node)
    
    # The longest path in the tree is the diameter
    diameter = max(dist2.values())
    print(diameter)

if __name__ == "__main__":
    main()