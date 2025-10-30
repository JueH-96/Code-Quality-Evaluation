import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    edges = []
    parent = list(range(n + 1))
    rank = [0] * (n + 1)

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j, weight):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1
            return weight
        return 0

    for _ in range(m):
        line = list(map(int, sys.stdin.readline().split()))
        k, c = line[0], line[1]
        subset = line[2:]
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((subset[i], subset[j], c))

    edges.sort(key=lambda x: x[2])
    mst_weight = 0
    num_edges = 0
    
    for u, v, weight in edges:
        mst_weight += union(u,v,weight)
        num_edges +=1

    num_connected_components = 0
    for i in range(1,n+1):
        if parent[i] == i:
            num_connected_components +=1

    if num_connected_components > 1:
        print(-1)
    else:
        print(mst_weight)

solve()