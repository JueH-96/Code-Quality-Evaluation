import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [sys.stdin.readline().strip() for _ in range(N)]
    edge_map = {s: i for i, s in enumerate(edges)}
    transition = [[] for _ in range(N)]
    for i in range(N):
        a, b = edges[i][0], edges[i][1]
        for j in range(N):
            if edges[j][0] == b:
                transition[i].append(j)
    
    # Build bipartite graph: left is 0..N-1, right is 0..N-1
    # Edge from u (right) to v (left)
    graph = [[] for _ in range(N)]
    for u in range(N):
        for v in transition[u]:
            graph[u].append(v)
    
    # Hopcroft-Karp algorithm
    def max_matching():
        pair_u = [-1] * N
        pair_v = [-1] * N
        dist = [0] * N
        
        def bfs():
            queue = deque()
            for u in range(N):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            dist_null = float('inf')
            while queue:
                u = queue.popleft()
                if dist[u] < dist_null:
                    for v in graph[u]:
                        if pair_v[v] == -1:
                            dist_null = dist[u] + 1
                        elif dist[pair_v[v]] == float('inf'):
                            dist[pair_v[v]] = dist[u] + 1
                            queue.append(pair_v[v])
            return dist_null != float('inf')
        
        def dfs(u):
            for v in graph[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        result = 0
        while bfs():
            for u in range(N):
                if pair_u[u] == -1:
                    if dfs(u):
                        result += 1
        return result
    
    matching = max_matching()
    print(N - matching)

if __name__ == "__main__":
    main()