from collections import defaultdict, deque

def solve():
    n = int(input())
    
    # Build adjacency list
    adj = defaultdict(list)
    degree = [0] * (n + 1)
    
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    # Find vertices with degree 2
    degree2_vertices = []
    for i in range(1, n + 1):
        if degree[i] == 2:
            degree2_vertices.append(i)
    
    def find_path(start, end):
        if start == end:
            return [start]
        
        visited = set()
        parent = {}
        queue = deque([start])
        visited.add(start)
        parent[start] = None
        
        while queue:
            curr = queue.popleft()
            if curr == end:
                # Reconstruct path
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        
        return None
    
    count = 0
    
    # Check all pairs of degree-2 vertices
    for i in range(len(degree2_vertices)):
        for j in range(i + 1, len(degree2_vertices)):
            u, v = degree2_vertices[i], degree2_vertices[j]
            
            # Find path between u and v
            path = find_path(u, v)
            
            if path is None:
                continue
            
            # Check if all vertices in the path have degree 2
            valid = True
            for vertex in path:
                if degree[vertex] != 2:
                    valid = False
                    break
            
            if valid:
                count += 1
    
    print(count)

solve()