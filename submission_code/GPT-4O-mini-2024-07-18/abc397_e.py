def can_decompose_tree(N, K, edges):
    from collections import defaultdict, deque
    
    # Create an adjacency list for the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Check the degree of each vertex
    degree_count = [0] * (N * K + 1)
    for u, v in edges:
        degree_count[u] += 1
        degree_count[v] += 1
    
    # Each path of length K has K+1 vertices, hence we need N paths
    # Each vertex in the path contributes to the degree of its endpoints
    # Hence, the degree of each vertex must be even or odd depending on the number of paths
    for i in range(1, N * K + 1):
        if degree_count[i] % 2 != 0:
            return "No"
    
    # Now we need to check if we can form exactly N paths of length K
    # Each path of length K uses K edges, hence we need N * K edges in total
    # We can use a BFS or DFS to try to form these paths
    visited = [False] * (N * K + 1)
    
    def bfs(start):
        queue = deque([start])
        path = []
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            path.append(node)
            if len(path) == K + 1:  # We need K edges, hence K + 1 nodes
                return path
            
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return path
    
    # Try to find N paths
    for i in range(1, N * K + 1):
        if not visited[i]:
            path = bfs(i)
            if len(path) != K + 1:
                return "No"
    
    return "Yes"

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, K = map(int, data[0].split())
    edges = [tuple(map(int, line.split())) for line in data[1:]]
    
    result = can_decompose_tree(N, K, edges)
    print(result)

if __name__ == "__main__":
    main()