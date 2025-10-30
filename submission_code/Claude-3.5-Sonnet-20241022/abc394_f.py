from collections import defaultdict, deque

def find_max_alkane(N, edges):
    # Build adjacency list
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    def get_subgraph_size(center, used_vertices):
        # BFS to find connected component size
        queue = deque([center])
        visited = {center}
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if u not in visited and u in used_vertices:
                    visited.add(u)
                    queue.append(u)
        return len(visited)
    
    def try_build_alkane(center):
        # Try to build alkane with this vertex as degree 4
        if len(adj[center]) < 4:
            return -1
            
        # Get all neighbors and their subtrees
        neighbors = []
        for neighbor in adj[center]:
            queue = deque([(neighbor, center)])
            subtree = {neighbor}
            while queue:
                v, parent = queue.popleft()
                for u in adj[v]:
                    if u != parent:
                        subtree.add(u)
                        queue.append((u, v))
            neighbors.append((len(subtree), subtree, neighbor))
        
        # Sort by subtree size in descending order
        neighbors.sort(reverse=True)
        
        # Must use exactly 4 neighbors
        if len(neighbors) < 4:
            return -1
            
        # Use the 4 largest subtrees
        used_vertices = {center}
        for _, subtree, _ in neighbors[:4]:
            used_vertices.update(subtree)
            
        # Check degrees of all vertices in the subgraph
        for v in used_vertices:
            degree = sum(1 for u in adj[v] if u in used_vertices)
            if degree != 1 and degree != 4:
                return -1
                
        # Verify it's connected
        size = get_subgraph_size(center, used_vertices)
        if size != len(used_vertices):
            return -1
            
        return len(used_vertices)
    
    # Try each vertex as the center
    max_size = -1
    for v in range(1, N+1):
        size = try_build_alkane(v)
        max_size = max(max_size, size)
    
    return max_size

# Read input
N = int(input())
edges = []
for _ in range(N-1):
    a, b = map(int, input().split())
    edges.append((a, b))

# Solve and output
print(find_max_alkane(N, edges))