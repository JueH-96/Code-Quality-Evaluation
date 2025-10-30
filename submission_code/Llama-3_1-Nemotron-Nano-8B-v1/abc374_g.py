import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    S = input[1:n+1]
    graph = [[] for _ in range(n)]
    for i in range(n):
        u = S[i][1]
        for j in range(n):
            if S[j][0] == u:
                graph[i].append(j)
    
    # Hopcroft-Karp algorithm for bipartite matching
    def hopcroft_karp():
        pair_U = [-1] * n
        pair_V = [-1] * n
        dist = [0] * n
        
        def bfs():
            queue = deque()
            for u in range(n):
                if pair_U[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            dist_null = float('inf')
            while queue:
                u = queue.popleft()
                if dist[u] < dist_null:
                    for v in graph[u]:
                        if pair_V[v] == -1:
                            dist_null = dist[u] + 1
                        elif dist[pair_V[v]] == float('inf'):
                            dist[pair_V[v]] = dist[u] + 1
                            queue.append(pair_V[v])
            return dist_null != float('inf')
        
        def dfs(u):
            for v in graph[u]:
                if pair_V[v] == -1 or (dist[pair_V[v]] == dist[u] + 1 and dfs(pair_V[v])):
                    pair_U[u] = v
                    pair_V[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        result = 0
        while bfs():
            for u in range(n):
                if pair_U[u] == -1:
                    if dfs(u):
                        result += 1
        return result
    
    max_matching = hopcroft_karp()
    print(n - max_matching)

if __name__ == "__main__":
    main()