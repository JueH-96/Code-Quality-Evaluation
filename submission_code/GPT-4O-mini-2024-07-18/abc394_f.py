def find_max_alkane_subgraph(n, edges):
    from collections import defaultdict, deque
    
    if n < 5:
        return -1  # An alkane must have at least one vertex of degree 4 and at least 5 vertices total.

    # Build the adjacency list for the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Count the degree of each vertex
    degree = {i: 0 for i in range(1, n + 1)}
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # Check if there is at least one vertex of degree 4
    if not any(d == 4 for d in degree.values()):
        return -1

    # We will use BFS to find the maximum alkane subgraph
    visited = set()
    max_alkane_size = 0

    def bfs(start):
        queue = deque([start])
        visited.add(start)
        alkane_vertices = [start]
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if degree[neighbor] == 1 or degree[neighbor] == 4:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        alkane_vertices.append(neighbor)
        
        # Check if we have at least one degree 4 vertex
        if any(degree[v] == 4 for v in alkane_vertices):
            return len(alkane_vertices)
        return 0

    # Start BFS from each unvisited node
    for i in range(1, n + 1):
        if i not in visited:
            max_alkane_size = max(max_alkane_size, bfs(i))

    return max_alkane_size if max_alkane_size > 0 else -1

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    n = int(data[0])
    edges = [tuple(map(int, line.split())) for line in data[1:]]
    
    result = find_max_alkane_subgraph(n, edges)
    print(result)

if __name__ == "__main__":
    main()