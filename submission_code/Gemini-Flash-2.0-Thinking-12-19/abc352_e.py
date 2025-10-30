import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    min_edges = {}
    for _ in range(m):
        line = list(map(int, sys.stdin.readline().split()))
        k = line[0]
        c = line[1]
        vertices = line[2:]
        for i in range(k):
            for j in range(i + 1, k):
                u, v = sorted([vertices[i], vertices[j]])
                edge = (u, v)
                if edge in min_edges:
                    min_edges[edge] = min(min_edges[edge], c)
                else:
                    min_edges[edge] = c
                    
    edges = []
    for (u, v), weight in min_edges.items():
        edges.append((weight, u, v))
        
    edges.sort(key=lambda x: x[0])
    
    parent = list(range(n + 1))
    
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]
        
    def union_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent[b] = a
            return True
        return False
        
    mst_weight = 0
    edges_in_mst = 0
    
    for weight, u, v in edges:
        if union_sets(u, v):
            mst_weight += weight
            edges_in_mst += 1
            
    if edges_in_mst == n - 1:
        print(mst_weight)
    else:
        print("-1")

if __name__ == '__main__':
    solve()