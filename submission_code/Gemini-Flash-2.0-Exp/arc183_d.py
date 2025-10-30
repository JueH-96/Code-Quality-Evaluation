def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def distance(start, end):
        q = [(start, 0)]
        visited = {start}
        while q:
            node, dist = q.pop(0)
            if node == end:
                return dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
        return -1

    leaves = []
    for i in range(1, n + 1):
        if len(adj[i]) == 1:
            leaves.append(i)

    result = []
    removed = [False] * (n + 1)

    def is_perfect_matching_possible(remaining_adj):
        if not remaining_adj:
            return True

        nodes = list(remaining_adj.keys())
        if not nodes:
            return True
        
        if len(nodes) % 2 != 0:
            return False

        import itertools
        
        for perm in itertools.permutations(nodes):
            matching = True
            for i in range(0, len(perm), 2):
                u = perm[i]
                v = perm[i+1]
                if v not in remaining_adj[u]:
                    matching = False
                    break
            if matching:
                return True
        return False
    
    for _ in range(n // 2):
        best_u = -1
        best_v = -1
        max_dist = -1

        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                u = leaves[i]
                v = leaves[j]

                if removed[u] or removed[v]:
                    continue

                temp_adj = {}
                for k in range(1, n + 1):
                    if not removed[k] and k != u and k != v:
                        temp_adj[k] = []
                        for neighbor in adj[k]:
                            if not removed[neighbor] and neighbor != u and neighbor != v:
                                temp_adj[k].append(neighbor)
                
                if is_perfect_matching_possible(temp_adj):
                    dist = distance(u, v)
                    if dist > max_dist:
                        max_dist = dist
                        best_u = u
                        best_v = v

        result.append((best_u, best_v))
        removed[best_u] = True
        removed[best_v] = True

        new_leaves = []
        for i in range(1, n + 1):
            if not removed[i]:
                count = 0
                for neighbor in adj[i]:
                    if not removed[neighbor]:
                        count += 1
                if count == 1:
                    new_leaves.append(i)
        leaves = new_leaves

    for u, v in result:
        print(u, v)

solve()