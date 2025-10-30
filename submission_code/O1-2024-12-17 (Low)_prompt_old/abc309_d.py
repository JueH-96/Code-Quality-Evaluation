def solve():
    import sys
    from collections import deque

    input = sys.stdin.readline
    
    N1, N2, M = map(int, input().split())
    N = N1 + N2

    # Build adjacency list
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    # BFS from vertex 1 to get distances dist1[]
    dist1 = [-1] * (N + 1)
    dist1[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for nxt in graph[u]:
            if dist1[nxt] == -1:
                dist1[nxt] = dist1[u] + 1
                queue.append(nxt)
    
    # BFS from vertex N (which is N1+N2) to get distances dist2[]
    dist2 = [-1] * (N + 1)
    dist2[N] = 0
    queue = deque([N])
    while queue:
        u = queue.popleft()
        for nxt in graph[u]:
            if dist2[nxt] == -1:
                dist2[nxt] = dist2[u] + 1
                queue.append(nxt)

    # We want to choose u in [1..N1] and v in [N1+1..N] to maximize:
    # dist1[u] + 1 + dist2[v].
    # Since we can choose any valid u and v, the maximum is:
    # (max over u in [1..N1] of dist1[u]) + (max over v in [N1+1..N] of dist2[v]) + 1
    max_dist1 = max(dist1[u] for u in range(1, N1 + 1))
    max_dist2 = max(dist2[v] for v in range(N1 + 1, N + 1))

    print(max_dist1 + max_dist2 + 1)

# Call solve()
if __name__ == "__main__":
    solve()