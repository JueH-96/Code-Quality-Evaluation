import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    adj = [[] for _ in range(N + 1)]
    deg = [0] * (N + 1)
    
    for _ in range(N - 1):
        u = int(input[ptr])
        v = int(input[ptr + 1])
        ptr += 2
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
    
    visited = [False] * (N + 1)
    answer = 0
    
    for u in range(1, N + 1):
        if deg[u] == 2 and not visited[u]:
            cluster = []
            q = deque()
            q.append((u, 0))
            visited[u] = True
            cluster.append(u)
            
            while q:
                v, dist = q.popleft()
                for w in adj[v]:
                    if not visited[w]:
                        visited[w] = True
                        if deg[w] == 3:
                            q.append((w, dist + 1))
                        elif deg[w] == 2:
                            if dist + 1 >= 2:
                                cluster.append(w)
            k = len(cluster)
            answer += k * (k - 1) // 2
    
    print(answer)

if __name__ == "__main__":
    main()