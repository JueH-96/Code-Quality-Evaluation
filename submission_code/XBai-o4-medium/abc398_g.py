import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * (N + 1)
    S = 0
    
    for start in range(1, N + 1):
        if not visited[start]:
            color = [-1] * (N + 1)
            q = deque()
            q.append(start)
            color[start] = 0
            visited[start] = True
            cnt0 = 1
            cnt1 = 0
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        color[v] = color[u] ^ 1
                        if color[v] == 0:
                            cnt0 += 1
                        else:
                            cnt1 += 1
                        q.append(v)
            
            d = cnt0 - cnt1
            S += d
    
    if N % 2 == 1:
        parity_AB = 0
    else:
        y = (N - S) // 2
        parity_AB = y % 2
    
    parity_M = M % 2
    total_parity = (parity_AB + parity_M) % 2
    
    if total_parity == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == "__main__":
    main()