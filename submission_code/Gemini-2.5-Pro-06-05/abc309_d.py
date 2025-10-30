import sys
from collections import deque

def solve():
    """
    Reads graph, performs two BFS traversals, and calculates the maximum shortest path.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read graph parameters
    try:
        N1, N2, M = map(int, input().split())
    except (IOError, ValueError):
        # Handle empty input case in some environments
        return
        
    N = N1 + N2
    
    # Adjacency list representation of the graph
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # BFS function to compute shortest paths from a source node in an unweighted graph
    def bfs(start_node):
        distances = [-1] * (N + 1)
        q = deque([start_node])
        distances[start_node] = 0
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    q.append(v)
        return distances

    # 1. Compute shortest paths from vertex 1 to all nodes in its component
    dist_from_1 = bfs(1)
    
    # 2. Compute shortest paths from vertex N to all nodes in its component
    dist_from_N = bfs(N)
    
    # 3. Find the maximum distance from vertex 1 to any node in its component (vertices 1 to N1)
    # This corresponds to choosing the optimal vertex `u`.
    max_d1 = max(dist_from_1[i] for i in range(1, N1 + 1))
    
    # 4. Find the maximum distance from vertex N to any node in its component (vertices N1+1 to N)
    # This corresponds to choosing the optimal vertex `v`.
    max_d2 = max(dist_from_N[i] for i in range(N1 + 1, N + 1))
    
    # 5. The maximum possible shortest path is the sum of these maximum distances plus 1 for the new edge.
    result = max_d1 + max_d2 + 1
    print(result)

solve()