import collections

def solve():
    n, k = map(int, input().split())
    if n == 1:
        if k >= 1:
            print(1)
        else:
            print(1)
        return
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    v_set = list(map(int, input().split()))
    
    if k == 0:
        print(1)
        return
    if k == 1:
        print(1)
        return
        
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def get_path(u, v):
        queue = collections.deque([(u, [u])])
        visited = {u}
        while queue:
            current_node, path = queue.popleft()
            if current_node == v:
                return path
            for neighbor in adj[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    required_vertices = set()
    for i in range(k):
        for j in range(i + 1, k):
            u = v_set[i]
            v = v_set[j]
            path_vertices = get_path(u, v)
            for vertex in path_vertices:
                required_vertices.add(vertex)
                
    if k == 1:
        required_vertices.add(v_set[0])

    print(len(required_vertices))

if __name__ == '__main__':
    solve()