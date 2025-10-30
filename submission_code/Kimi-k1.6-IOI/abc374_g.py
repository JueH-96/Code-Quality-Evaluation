import sys
from collections import deque

def hopcroft_karp(graph, n_U, n_V):
    pair_U = [-1] * n_U
    pair_V = [-1] * n_V
    dist = [0] * n_U

    def bfs():
        queue = deque()
        for u in range(n_U):
            if pair_U[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        dist_found = float('inf')
        while queue:
            u = queue.popleft()
            if dist[u] < dist_found:
                for v in graph[u]:
                    if pair_V[v] == -1:
                        dist_found = dist[u] + 1
                    elif dist[pair_V[v]] == float('inf'):
                        dist[pair_V[v]] = dist[u] + 1
                        queue.append(pair_V[v])
        return dist_found != float('inf')

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
        for u in range(n_U):
            if pair_U[u] == -1:
                if dfs(u):
                    result += 1
    return result

def main():
    n = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(n)]
    
    bipartite = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if S[u][1] == S[v][0]:
                bipartite[u].append(v)
    
    M = hopcroft_karp(bipartite, n, n)
    print(n - M)

if __name__ == "__main__":
    main()