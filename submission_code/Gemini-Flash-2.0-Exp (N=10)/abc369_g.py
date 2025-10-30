def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        edges.append((u, v, l))

    adj = [[] for _ in range(n + 1)]
    for u, v, l in edges:
        adj[u].append((v, l))
        adj[v].append((u, l))

    def get_path_length(path):
        length = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            for neighbor, l in adj[u]:
                if neighbor == v:
                    length += l
                    break
        return length

    def find_path(start, end, visited, current_path):
        visited[start] = True
        current_path.append(start)
        if start == end:
            return current_path
        
        for neighbor, _ in adj[start]:
            if not visited[neighbor]:
                path = find_path(neighbor, end, visited.copy(), current_path.copy())
                if path:
                    return path
        return None

    def calculate_min_walk_length(required_vertices):
        if not required_vertices:
            return 0
        
        min_length = float('inf')
        import itertools
        for perm in itertools.permutations(required_vertices):
            current_path = [1]
            for v in perm:
                path_to_v = find_path(current_path[-1], v, [False] * (n+1), [])
                current_path.extend(path_to_v[1:])
            path_to_1 = find_path(current_path[-1], 1, [False] * (n+1), [])
            current_path.extend(path_to_1[1:])
            min_length = min(min_length, get_path_length(current_path))
        return min_length

    
    for k in range(1, n + 1):
        max_min_length = 0
        import itertools
        for aoki_vertices in itertools.combinations(range(1, n + 1), k):
            min_length = calculate_min_walk_length(list(aoki_vertices))
            max_min_length = max(max_min_length, min_length)
        print(max_min_length)

solve()