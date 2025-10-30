import sys
from collections import deque

def hopcroft_karp(graph, U, V):
    pair_U = [-1] * U
    pair_V = [-1] * V
    dist = [0] * U
    result = 0

    def bfs():
        queue = deque()
        for u in range(U):
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
    
    while bfs():
        for u in range(U):
            if pair_U[u] == -1:
                if dfs(u):
                    result += 1
    return result

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    s_list = input[1:n+1]
    
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if s_list[i][1] == s_list[j][0]:
                adj[i].append(j)
    
    max_matching = hopcroft_karp(adj, n, n)
    print(n - max_matching)

if __name__ == "__main__":
    main()