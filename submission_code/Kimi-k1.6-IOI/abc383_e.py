import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1

    edges = []
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        w = int(input[ptr])
        ptr += 1
        edges.append((u, v, w))
    
    A = list(map(int, input[ptr:ptr+K]))
    ptr += K
    B = list(map(int, input[ptr:ptr+K]))
    ptr += K

    # Compute frequency arrays for A and B
    a = [0] * (N + 1)
    for x in A:
        a[x] += 1
    b = [0] * (N + 1)
    for x in B:
        b[x] += 1

    # Kruskal's algorithm to find MST edges
    edges.sort(key=lambda x: x[2])

    class DSUKruskal:
        def __init__(self, size):
            self.parent = list(range(size + 1))
            self.rank = [0] * (size + 1)
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return False
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1
            return True
    
    dsu_kruskal = DSUKruskal(N)
    mst_edges = []
    for edge in edges:
        u, v, w = edge
        if dsu_kruskal.find(u) != dsu_kruskal.find(v):
            dsu_kruskal.union(u, v)
            mst_edges.append(edge)
    
    # Sort MST edges by weight again to ensure correct processing order
    mst_edges.sort(key=lambda x: x[2])

    # DSU to track a and b counts
    class DSUCount:
        def __init__(self, size, a, b):
            self.parent = list(range(size + 1))
            self.a_count = a.copy()
            self.b_count = b.copy()
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return False
            self.parent[y_root] = x_root
            self.a_count[x_root] += self.a_count[y_root]
            self.b_count[x_root] += self.b_count[y_root]
            return True
    
    dsu_count = DSUCount(N, a, b)
    total_sum = 0

    for u, v, w in mst_edges:
        root_u = dsu_count.find(u)
        root_v = dsu_count.find(v)
        if root_u != root_v:
            a1 = dsu_count.a_count[root_u]
            b1 = dsu_count.b_count[root_u]
            a2 = dsu_count.a_count[root_v]
            b2 = dsu_count.b_count[root_v]
            old = min(a1, b1) + min(a2, b2)
            dsu_count.union(u, v)
            new_root = dsu_count.find(u)
            new_a = dsu_count.a_count[new_root]
            new_b = dsu_count.b_count[new_root]
            new_min = min(new_a, new_b)
            total_sum += (new_min - old) * w
    
    print(total_sum)

if __name__ == "__main__":
    main()