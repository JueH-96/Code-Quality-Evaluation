import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    operations = []
    for i in range(1, q + 1):
        l, r, c = map(int, sys.stdin.readline().split())
        operations.append({'l': l, 'r': r, 'c': c, 'index': i})
    
    operations.sort(key=lambda op: op['c'])
    
    parent = list(range(n + q + 1))
    rank = [0] * (n + q + 1)
    
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]
        
    def unite_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            if rank[a] < rank[b]:
                a, b = b, a
            parent[b] = a
            if rank[a] == rank[b]:
                rank[a] += 1
            return True
        return False
        
    mst_cost = 0
    edges_in_mst = 0
    
    for operation in operations:
        l_i, r_i, c_i, index_i = operation['l'], operation['r'], operation['c'], operation['index']
        for j in range(l_i, r_i + 1):
            u = n + index_i
            v = j
            if unite_sets(u, v):
                mst_cost += c_i
                edges_in_mst += 1
                
    if edges_in_mst == n + q - 1:
        print(mst_cost)
    else:
        print("-1")

if __name__ == '__main__':
    solve()