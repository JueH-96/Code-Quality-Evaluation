import collections

def solve():
    n, m = map(int, input().split())
    operations = []
    for _ in range(m):
        line = list(map(int, input().split()))
        k = line[0]
        c = line[1]
        vertices_set = tuple(sorted(line[2:]))
        operations.append({'vertices': vertices_set, 'weight': c})
    
    edge_weights = collections.defaultdict(lambda: float('inf'))
    for op in operations:
        vertices = op['vertices']
        weight = op['weight']
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = min(vertices[i], vertices[j]), max(vertices[i], vertices[j])
                edge_weights[(u, v)] = min(edge_weights[(u, v)], weight)
                
    edges = []
    for (u, v), weight in edge_weights.items():
        edges.append({'u': u, 'v': v, 'weight': weight})
        
    edges.sort(key=lambda e: e['weight'])
    
    parent = list(range(n + 1))
    
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]
        
    def unite_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent[b] = a
            return True
        return False
        
    mst_weight = 0
    edges_in_mst = 0
    
    for edge_info in edges:
        u, v, weight = edge_info['u'], edge_info['v'], edge_info['weight']
        if unite_sets(u, v):
            mst_weight += weight
            edges_in_mst += 1
            if edges_in_mst == n - 1:
                break
                
    if edges_in_mst < n - 1:
        print("-1")
    else:
        print(mst_weight)

if __name__ == '__main__':
    solve()