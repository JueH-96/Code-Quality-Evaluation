import collections

def solve():
    n, k = map(int, input().split())
    if n == 1:
        input() # edge line, but no edges
        v_line = list(map(int, input().split()))
        if k == 0:
            print(0)
        else:
            print(1)
        return
        
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    given_vertices = list(map(int, input().split()))
    
    if k <= 1:
        print(k)
        return
        
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def get_path(u, v):
        parent = {}
        visited = set()
        queue = collections.deque([u])
        visited.add(u)
        parent[u] = None
        path_found = False
        while queue:
            current_node = queue.popleft()
            if current_node == v:
                path_found = True
                break
            for neighbor in adj[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current_node
                    queue.append(neighbor)
        if not path_found:
            return []
        path = []
        current = v
        while current is not None:
            path.append(current)
            current = parent.get(current)
        return path[::-1]
        
    required_vertices = set()
    for i in range(k):
        for j in range(i + 1, k):
            u = given_vertices[i]
            v = given_vertices[j]
            path_vertices = get_path(u, v)
            for vertex in path_vertices:
                required_vertices.add(vertex)
                
    print(len(required_vertices))

if __name__ == '__main__':
    solve()