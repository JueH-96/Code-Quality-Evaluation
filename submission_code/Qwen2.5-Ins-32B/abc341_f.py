import sys
from collections import deque

def solve():
    input = sys.stdin.read
    N, M, *UV, *W, *A = map(int, input().split())
    graph = [[] for _ in range(N)]
    for u, v in zip(UV[::2], UV[1::2]):
        graph[u-1].append(v-1)
        graph[v-1].append(u-1)
    
    def bfs(start):
        queue = deque([start])
        visited = [False] * N
        visited[start] = True
        while queue:
            v = queue.popleft()
            for nv in graph[v]:
                if not visited[nv] and W[nv] < W[v]:
                    visited[nv] = True
                    queue.append(nv)
        return visited
    
    ans = 0
    for i in range(N):
        if A[i] > 0:
            reachable = bfs(i)
            total_weight = sum(W[j] for j in range(N) if reachable[j])
            ans += A[i] * (W[i] - total_weight)
    
    print(ans)

solve()