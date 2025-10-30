def main():
    import sys
    from collections import deque
    
    input = sys.stdin.readline

    N1, N2, M = map(int, input().split())
    n = N1 + N2
    
    # Adjacency list for the entire graph
    adj = [[] for _ in range(n + 1)]
    
    for _ in range(M):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # BFS from vertex 1
    dist1 = [-1] * (n + 1)
    dist1[1] = 0
    queue = deque([1])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if dist1[nxt] == -1:
                dist1[nxt] = dist1[node] + 1
                queue.append(nxt)
    
    # BFS from vertex (N1 + N2)
    dist2 = [-1] * (n + 1)
    dist2[n] = 0
    queue = deque([n])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if dist2[nxt] == -1:
                dist2[nxt] = dist2[node] + 1
                queue.append(nxt)
    
    # max distance from 1 to any vertex u in [1..N1]
    x = max(dist1[u] for u in range(1, N1 + 1))
    # max distance from (N1+N2) to any vertex v in [N1+1..N1+N2]
    y = max(dist2[v] for v in range(N1 + 1, n + 1))
    
    # The answer is (x + 1 + y)
    print(x + 1 + y)

# Do not forget to call main()
if __name__ == "__main__":
    main()