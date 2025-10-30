import heapq

def read_input():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        U, V, B = map(int, input().split())
        graph[U].append((V, B))
        graph[V].append((U, B))
    return N, M, A, graph

def solve():
    N, M, A, graph = read_input()
    min_weights = [float('inf')] * (N+1)
    min_weights[1] = 0
    visited = [False] * (N+1)
    heap = [(0, 1)]
    while heap:
        weight, node = heapq.heappop(heap)
        if visited[node]:
            continue
        visited[node] = True
        for neighbor, edge_weight in graph[node]:
            if not visited[neighbor] and min_weights[node] + edge_weight + A[node-1] < min_weights[neighbor]:
                min_weights[neighbor] = min_weights[node] + edge_weight + A[node-1]
                heapq.heappush(heap, (min_weights[neighbor], neighbor))
    return min_weights[2:]

print(*solve())