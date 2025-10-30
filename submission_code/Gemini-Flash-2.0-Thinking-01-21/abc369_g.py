import collections

def solve():
    n = int(input())
    if n == 2:
        u1, v1, l1 = map(int, input().split())
        total_length = l1
        print(2 * total_length)
        print(2 * total_length)
        return
        
    edges = []
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        edges.append(((u, v), l))
    
    adjacency_list = collections.defaultdict(list)
    total_edge_length = 0
    for (u, v), length in edges:
        adjacency_list[u].append((v, length))
        adjacency_list[v].append((u, length))
        total_edge_length += length
        
    def get_distance(start_node):
        distances = {}
        for i in range(1, n + 1):
            distances[i] = float('inf')
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            u = queue.popleft()
            for v, length in adjacency_list[u]:
                if distances[v] == float('inf'):
                    distances[v] = distances[u] + length
                    queue.append(v)
        return distances
        
    distances_from_1 = get_distance(1)
    
    def get_path(u, v, parent_map):
        path = []
        curr = v
        while curr != u:
            path.append((parent_map[curr], curr))
            curr = parent_map[curr]
        return path
        
    def get_subtree_length(target_vertices):
        if not target_vertices:
            return 0
        minimal_subtree_edges = set()
        first_vertex = target_vertices[0]
        distances_bfs_result = {}
        parent_map = {}
        
        for i in range(1, n + 1):
            distances_bfs_result[i] = float('inf')
        distances_bfs_result[1] = 0
        parent_map[1] = None
        queue = collections.deque([1])
        visited = {1}
        
        while queue:
            u = queue.popleft()
            for v, length in adjacency_list[u]:
                if v not in visited:
                    visited.add(v)
                    distances_bfs_result[v] = distances_bfs_result[u] + length
                    parent_map[v] = u
                    queue.append(v)
                    
        paths_edges = set()
        for v in target_vertices:
            current_v = v
            while current_v != 1:
                u = parent_map[current_v]
                edge = tuple(sorted((u, current_v)))
                paths_edges.add(edge)
                current_v = u
                
        subtree_length = 0
        for (u, v), length in edges:
            edge_uv = tuple(sorted((u, v)))
            if edge_uv in paths_edges:
                subtree_length += length
                
        return subtree_length

    results = []
    for k in range(1, n + 1):
        if k >= 3:
            results.append(2 * total_edge_length)
        elif k == 1:
            max_dist = 0
            for i in range(1, n + 1):
                if i != 1:
                    max_dist = max(max_dist, distances_from_1[i])
            results.append(2 * max_dist)
        elif k == 2:
            max_subtree_len = 0
            import itertools
            candidate_vertices = list(range(1, n + 1))
            vertex_pairs = itertools.combinations(candidate_vertices, 2)
            for v1, v2 in vertex_pairs:
                subtree_len = get_subtree_length([v1, v2])
                max_subtree_len = max(max_subtree_len, subtree_len)
            results.append(2 * max_subtree_len)
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()