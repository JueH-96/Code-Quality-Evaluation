from collections import defaultdict, deque

def solve():
    n = int(input())
    
    # Build the graph
    graph = defaultdict(list)
    
    # Read all edges
    for i in range(n-1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    # Current vertices in the tree
    vertices = set(range(1, n+1))
    result = []
    
    def get_distance(u, v):
        queue = deque([(u, 0)])
        visited = {u}
        
        while queue:
            node, dist = queue.popleft()
            if node == v:
                return dist
            
            for neighbor in graph[node]:
                if neighbor in visited or neighbor not in vertices:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
        
        return -1
    
    def get_leaves():
        leaves = []
        for v in vertices:
            degree = sum(1 for neighbor in graph[v] if neighbor in vertices)
            if degree == 1:
                leaves.append(v)
        return leaves
    
    def is_connected_after_removal(u, v):
        remaining = vertices - {u, v}
        if len(remaining) <= 1:
            return True
        
        start = next(iter(remaining))
        visited = {start}
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in remaining and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(remaining)
    
    for _ in range(n//2):
        leaves = get_leaves()
        
        # Find the pair of leaves with maximum distance such that removal is valid
        best_pair = None
        best_distance = -1
        
        for i in range(len(leaves)):
            for j in range(i+1, len(leaves)):
                u, v = leaves[i], leaves[j]
                remaining_count = len(vertices) - 2
                # Check if remaining graph has even vertices and stays connected
                if remaining_count % 2 == 0 and is_connected_after_removal(u, v):
                    dist = get_distance(u, v)
                    if dist > best_distance:
                        best_distance = dist
                        best_pair = (u, v)
        
        if best_pair:
            u, v = best_pair
            result.append((u, v))
            vertices.remove(u)
            vertices.remove(v)
    
    for u, v in result:
        print(u, v)

solve()