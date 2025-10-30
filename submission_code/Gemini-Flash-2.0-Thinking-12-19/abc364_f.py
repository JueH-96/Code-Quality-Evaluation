import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    operations = []
    for i in range(q):
        l, r, c = map(int, sys.stdin.readline().split())
        operations.append({'l': l, 'r': r, 'c': c, 'index': i + 1})
    
    operations.sort(key=lambda op: op['c'])
    
    parent = list(range(n + q + 1))
    
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
        
    mst_cost = 0
    edges_count = 0
    
    for operation in operations:
        l, r, c, index = operation['l'], operation['r'], operation['c'], operation['index']
        vertex_n_plus_i = n + index
        for j in range(l, r + 1):
            if unite_sets(j, vertex_n_plus_i):
                mst_cost += c
                edges_count += 1
                
    if edges_count == n + q - 1:
        print(mst_cost)
    else:
        print("-1")

if __name__ == '__main__':
    solve()