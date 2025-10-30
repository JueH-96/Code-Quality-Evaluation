from collections import deque
import sys

def bfs(start, adj, n):
    dist = [-1] * (n+1)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def main():
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    
    # Build adjacency lists
    adj = [[] for _ in range(N+1)]
    adj_rev = [[] for _ in range(N+1)]
    
    idx = 2
    for _ in range(M):
        a, b = map(int, input_data[idx:idx+2])
        idx += 2
        adj[a].append(b)
        adj_rev[b].append(a)
    
    # BFS from 1 in the original graph
    dist1 = bfs(1, adj, N)
    # BFS from 1 in the reversed graph
    dist2 = bfs(1, adj_rev, N)
    
    # Find the minimum dist1[x] + dist2[x] for x where both are reachable
    ans = -1
    for x in range(1, N+1):
        if dist1[x] != -1 and dist2[x] != -1:
            cand = dist1[x] + dist2[x]
            if ans == -1 or cand < ans:
                ans = cand
    
    print(ans)

# Do not forget to call main
if __name__ == "__main__":
    main()