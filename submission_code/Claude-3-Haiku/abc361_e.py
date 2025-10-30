from collections import defaultdict
import heapq

def min_travel_distance(N, roads):
    graph = defaultdict(list)
    for a, b, c in roads:
        graph[a].append((b, c))
        graph[b].append((a, c))

    visited = set()
    heap = [(0, 1)]  # (distance, node)
    total_distance = 0

    while len(visited) < N:
        distance, node = heapq.heappop(heap)
        if node not in visited:
            visited.add(node)
            total_distance += distance

            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(heap, (weight, neighbor))

    return total_distance

# Read input
N = int(input())
roads = []
for _ in range(N - 1):
    a, b, c = map(int, input().split())
    roads.append((a, b, c))

# Solve the problem
result = min_travel_distance(N, roads)
print(result)