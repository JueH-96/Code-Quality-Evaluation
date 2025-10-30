import collections

def solve():
    n, m = map(int, input().split())
    relationships = []
    for _ in range(m):
        u, v, x, y = map(int, input().split())
        relationships.append(((u, v), (x, y)))
    
    coords = [None] * (n + 1)
    coords[1] = (0, 0)
    
    adj = collections.defaultdict(list)
    for (u, v), (x, y) in relationships:
        adj[u].append((v, (x, y)))
        
    queue = collections.deque([1])
    visited = {1}
    
    while queue:
        u = queue.popleft()
        current_coords = coords[u]
        if current_coords is None:
            continue
            
        for v, displacement in adj[u]:
            if v not in visited:
                x_disp, y_disp = displacement
                coords[v] = (current_coords[0] + x_disp, current_coords[1] + y_disp)
                visited.add(v)
                queue.append(v)
                
    result = []
    for i in range(1, n + 1):
        if coords[i] is not None:
            result.append(f"{coords[i][0]} {coords[i][1]}")
        else:
            result.append("undecidable")
            
    print('
'.join(result))

if __name__ == '__main__':
    solve()