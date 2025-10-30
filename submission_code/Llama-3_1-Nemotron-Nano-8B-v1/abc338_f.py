import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        edges.append((u, v, w))
    
    # Check strong connectivity
    adj = [[] for _ in range(N)]
    for u, v, _ in edges:
        adj[u].append(v)
    
    is_strong = True
    for u in range(N):
        visited = [False] * N
        q = deque([u])
        visited[u] = True
        while q:
            node = q.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)
        if not all(visited):
            is_strong = False
            break
    if not is_strong:
        print("No")
        return
    
    # Compute all-pairs shortest paths using Bellman-Ford
    INF = float('inf')
    dist = [[INF] * N for _ in range(N)]
    for u in range(N):
        dist[u][u] = 0
        for _ in range(N - 1):
            updated = False
            for a, b, w in edges:
                if dist[u][a] + w < dist[u][b]:
                    dist[u][b] = dist[u][a] + w
                    updated = True
            if not updated:
                break
    
    # Check if any distance is still INF (shouldn't happen as graph is strongly connected)
    for u in range(N):
        if any(d == INF for d in dist[u]):
            print("No")
            return
    
    # TSP DP
    size = 1 << N
    dp = [[INF] * N for _ in range(size)]
    for u in range(N):
        dp[1 << u][u] = 0
    
    for mask in range(size):
        for u in range(N):
            if not (mask & (1 << u)):
                continue
            # Try to go to v not in mask
            for v in range(N):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                    dp[new_mask][v] = dp[mask][u] + dist[u][v]
    
    full_mask = (1 << N) - 1
    ans = min(dp[full_mask][u] for u in range(N))
    print(ans if ans != INF else "No")

if __name__ == "__main__":
    main()