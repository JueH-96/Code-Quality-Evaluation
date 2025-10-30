import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    adj = [[] for _ in range(n)]
    total_sum = 0
    index = 1
    for _ in range(n-1):
        a = int(data[index]) - 1
        b = int(data[index+1]) - 1
        c = int(data[index+2])
        index += 3
        adj[a].append((b, c))
        adj[b].append((a, c))
        total_sum += c
    
    if n == 1:
        print(0)
        return
    
    def bfs(start, n, adj):
        dist = [-1] * n
        q = deque([start])
        dist[start] = 0
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + w
                    q.append(v)
        return dist
    
    dist0 = bfs(0, n, adj)
    u = max(range(n), key=lambda x: dist0[x])
    dist1 = bfs(u, n, adj)
    diam = max(dist1)
    
    ans = 2 * total_sum - diam
    print(ans)

if __name__ == "__main__":
    main()