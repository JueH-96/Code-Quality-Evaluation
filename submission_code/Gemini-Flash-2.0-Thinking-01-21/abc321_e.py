def solve():
    n, x, k = map(int, input().split())
    
    def get_parent(v):
        if v > 1:
            return v // 2
        return 0
        
    def get_children(v, current_n):
        children = []
        child1 = 2 * v
        if child1 <= current_n:
            children.append(child1)
        child2 = 2 * v + 1
        if child2 <= current_n:
            children.append(child2)
        return children
        
    def count_vertices_at_distance(start_vertex, target_distance, n_vertices):
        if target_distance < 0:
            return 0
        if target_distance == 0:
            return 1
            
        count = 0
        queue = [(start_vertex, 0)]
        visited = {start_vertex}
        vertices_at_k_distance = set()
        
        while queue:
            current_vertex, current_distance = queue.pop(0)
            if current_distance == target_distance:
                vertices_at_k_distance.add(current_vertex)
                continue
            if current_distance > target_distance:
                continue
                
            # Explore parent
            parent = get_parent(current_vertex)
            if parent > 0 and parent <= n_vertices and parent not in visited:
                visited.add(parent)
                queue.append((parent, current_distance + 1))
                
            # Explore children
            children = get_children(current_vertex, n_vertices)
            for child in children:
                if child not in visited:
                    visited.add(child)
                    queue.append((child, current_distance + 1))
                    
        return len(vertices_at_k_distance)

    result = count_vertices_at_distance(x, k, n)
    print(result)

t = int(input())
for _ in range(t):
    solve()