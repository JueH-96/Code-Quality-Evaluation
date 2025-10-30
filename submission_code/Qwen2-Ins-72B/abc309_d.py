from collections import deque

def bfs(graph, start, n):
    dist = [-1] * (n + 1)
    dist[start] = 0
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for next_v in graph[v]:
            if dist[next_v] != -1:
                continue
            dist[next_v] = dist[v] + 1
            queue.append(next_v)
    return dist

def main():
    N1, N2, M = map(int, input().split())
    N = N1 + N2
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)

    dist1 = bfs(graph, 1, N)
    distN = bfs(graph, N, N)

    max_d = 0
    for i in range(1, N1 + 1):
        for j in range(N1 + 1, N + 1):
            if dist1[i] == -1 or distN[j] == -1:
                continue
            d = dist1[i] + 1 + distN[j]
            max_d = max(max_d, d)

    print(max_d)

if __name__ == "__main__":
    main()