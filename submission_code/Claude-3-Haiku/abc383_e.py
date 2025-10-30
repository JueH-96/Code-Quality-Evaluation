from collections import defaultdict
import heapq

def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances[end]

def solve():
    N, M, K = map(int, input().split())
    graph = defaultdict(dict)
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u][v] = w
        graph[v][u] = w
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    min_sum = float('inf')
    for perm in itertools.permutations(B):
        current_sum = 0
        for a, b in zip(A, perm):
            current_sum += dijkstra(graph, a, b)
        min_sum = min(min_sum, current_sum)

    print(min_sum)

solve()