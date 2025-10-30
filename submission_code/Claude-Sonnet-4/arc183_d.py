from collections import defaultdict, deque

def solve():
    n = int(input())
    
    # Build adjacency list
    adj = defaultdict(list)
    edges = []
    
    for i in range(n - 1):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
        edges.append((a, b))
    
    # Track degrees
    degree = defaultdict(int)
    for i in range(1, n + 1):
        degree[i] = len(adj[i])
    
    # Initial perfect matching pairs
    matched_pairs = set()
    for i in range(1, n // 2 + 1):
        matched_pairs.add((2 * i - 1, 2 * i))
    
    result = []
    remaining_vertices = set(range(1, n + 1))
    
    def bfs_distance(start, end, vertices, adjacency):
        if start == end:
            return 0
        if start not in vertices or end not in vertices:
            return -1
            
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in adjacency[node]:
                if neighbor in vertices and neighbor not in visited:
                    if neighbor == end:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1
    
    def can_remove_pair(v1, v2, vertices, adjacency):
        # Check if removing v1 and v2 leaves a graph with perfect matching
        remaining = vertices - {v1, v2}
        if len(remaining) % 2 != 0:
            return False
        
        # For this problem, we need to check if the remaining graph has a perfect matching
        # Since we start with a perfect matching and remove matched pairs, 
        # we need to ensure the remaining structure is valid
        
        # Simple heuristic: if both vertices are leaves and removing them doesn't disconnect
        # the remaining graph in a way that breaks perfect matching
        return True
    
    while len(remaining_vertices) > 0:
        # Find current leaves
        leaves = []
        for v in remaining_vertices:
            if degree[v] == 1:
                leaves.append(v)
        
        if len(leaves) < 2:
            # Find vertices with degree 2 or more if no leaves
            candidates = list(remaining_vertices)
            leaves = candidates[:2] if len(candidates) >= 2 else candidates
        
        # Find the best pair of leaves to remove
        best_pair = None
        best_distance = -1
        
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                v1, v2 = leaves[i], leaves[j]
                
                # Calculate distance between v1 and v2
                dist = bfs_distance(v1, v2, remaining_vertices, adj)
                
                if dist > best_distance:
                    best_distance = dist
                    best_pair = (v1, v2)
        
        if best_pair is None:
            # Fallback: just pick any two remaining vertices
            vertices_list = list(remaining_vertices)
            if len(vertices_list) >= 2:
                best_pair = (vertices_list[0], vertices_list[1])
            else:
                break
        
        # Remove the chosen pair
        v1, v2 = best_pair
        result.append((v1, v2))
        
        # Update remaining vertices
        remaining_vertices.remove(v1)
        remaining_vertices.remove(v2)
        
        # Update degrees of neighbors
        for neighbor in adj[v1]:
            if neighbor in remaining_vertices:
                degree[neighbor] -= 1
        
        for neighbor in adj[v2]:
            if neighbor in remaining_vertices:
                degree[neighbor] -= 1
    
    # Output result
    for pair in result:
        print(pair[0], pair[1])

solve()