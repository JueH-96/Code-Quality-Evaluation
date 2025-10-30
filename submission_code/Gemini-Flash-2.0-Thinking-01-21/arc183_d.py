import collections

def solve():
    n = int(input())
    edges_input = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges_input.append((u, v))
    
    current_edges = list(edges_input)
    current_vertices = set(range(1, n + 1))
    operations = []
    
    while len(current_vertices) > 0:
        if not current_vertices:
            break
        
        adj = collections.defaultdict(list)
        for u, v in current_edges:
            if u in current_vertices and v in current_vertices:
                adj[u].append(v)
                adj[v].append(u)
                
        leaves = []
        for v in current_vertices:
            if len(adj[v]) <= 1:
                leaves.append(v)
                
        if not leaves:
            break
            
        best_pair = None
        max_distance = -1
        
        import itertools
        leaf_pairs = list(itertools.combinations(leaves, 2))
        if not leaf_pairs and len(leaves) >= 2:
            leaf_pairs = [(leaves[0], leaves[1])]

        if not leaf_pairs:
            if len(leaves) >= 1:
                leaf1 = leaves[0]
                farthest_leaf = None
                max_dist_leaf = -1
                for leaf2 in leaves:
                    if leaf1 == leaf2:
                        continue
                    q = collections.deque([(leaf1, 0)])
                    visited = {leaf1}
                    distances = {}
                    distances[leaf1] = 0
                    while q:
                        u_node, dist = q.popleft()
                        if u_node == leaf2:
                            current_dist = dist
                            if current_dist > max_dist_leaf:
                                max_dist_leaf = current_dist
                                farthest_leaf = leaf2
                            break
                        for neighbor in adj[u_node]:
                            if neighbor not in visited and neighbor in current_vertices:
                                visited.add(neighbor)
                                distances[neighbor] = dist + 1
                                q.append((neighbor, dist + 1))
                if farthest_leaf:
                    best_pair = tuple(sorted((leaf1, farthest_leaf)))
                    max_distance = max_dist_leaf
                    
        else:
            for u_leaf, v_leaf in leaf_pairs:
                if u_leaf not in current_vertices or v_leaf not in current_vertices:
                    continue
                q = collections.deque([(u_leaf, 0)])
                visited = {u_leaf}
                distance = -1
                while q:
                    u_node, dist = q.popleft()
                    if u_node == v_leaf:
                        distance = dist
                        break
                    for neighbor in adj[u_node]:
                        if neighbor not in visited and neighbor in current_vertices:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))
                if distance > max_distance:
                    max_distance = distance
                    best_pair = tuple(sorted((u_leaf, v_leaf)))
                    
        if best_pair:
            u_remove, v_remove = best_pair
            operations.append(best_pair)
            current_vertices.remove(u_remove)
            current_vertices.remove(v_remove)
            next_edges = []
            for u, v in current_edges:
                if (u != u_remove and u != v_remove) and (v != u_remove and v != v_remove):
                    next_edges.append((u, v))
            current_edges = next_edges
        else:
            break

    for pair in operations:
        print(f"{pair[1]} {pair[0]}")

if __name__ == '__main__':
    solve()