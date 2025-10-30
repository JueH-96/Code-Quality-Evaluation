import sys
from collections import defaultdict, deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    for i in range(M):
        u = int(data[2 + 3 * i]) - 1
        v = int(data[3 + 3 * i]) - 1
        w = int(data[4 + 3 * i])
        edges.append((u, v, w))
    
    def dfs(node, visited):
        visited[node] = True
        for u, v, w in edges:
            if u == node and not visited[v]:
                dfs(v, visited)
    
    visited = [False] * N
    dfs(0, visited)
    
    if not all(visited):
        print("No")
        return
    
    def find_min_weight():
        dp = [[float('inf')] * N for _ in range(1 << N)]
        dp[1][0] = 0
        
        for mask in range(1 << N):
            for u in range(N):
                if dp[mask][u] == float('inf'):
                    continue
                for v, w in edges:
                    if u == v:
                        continue
                    if mask & (1 << v) == 0:
                        dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + w)
        
        min_weight = float('inf')
        for u in range(N):
            min_weight = min(min_weight, dp[(1 << N) - 1][u])
        
        return min_weight if min_weight != float('inf') else float('inf')
    
    min_weight = find_min_weight()
    
    if min_weight == float('inf'):
        print("No")
    else:
        print(min_weight)

if __name__ == "__main__":
    main()