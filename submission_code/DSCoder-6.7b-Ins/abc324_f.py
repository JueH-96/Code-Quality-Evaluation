import heapq

def solve():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        graph[u].append((v, b, c))

    # Dijkstra's algorithm
    dist = [float('inf')] * (N+1)
    dist[1] = 0
    heap = [(0, 1)]
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist[node]:
            continue
        for neighbor, b, c in graph[node]:
            if dist[node] + c < dist[neighbor]:
                dist[neighbor] = dist[node] + c
                heapq.heappush(heap, (dist[neighbor], neighbor))

    # Backtracking
    beauty = [0] * (N+1)
    cost = [0] * (N+1)
    visited = [False] * (N+1)
    stack = [N]
    while stack:
        node = stack[-1]
        if visited[node]:
            stack.pop()
            continue
        visited[node] = True
        for neighbor, b, c in graph[node]:
            if dist[node] - c == dist[neighbor]:
                beauty[node] += b
                cost[node] += c
                stack.append(neighbor)
                break

    # Calculate the maximum beauty ratio
    max_ratio = 0
    for node in range(1, N+1):
        if cost[node] != 0:
            max_ratio = max(max_ratio, beauty[node] / cost[node])
    print(max_ratio)

solve()