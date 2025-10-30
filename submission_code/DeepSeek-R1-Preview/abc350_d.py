class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size + 1))  # 1-based indexing
        self.rank = [0] * (size + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    edges = []
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        edges.append((u, v))
    
    uf = UnionFind(N)
    for u, v in edges:
        uf.union(u, v)
    
    from collections import defaultdict
    size = defaultdict(int)
    for node in range(1, N + 1):
        root = uf.find(node)
        size[root] += 1
    
    edge_count = defaultdict(int)
    for u, v in edges:
        root = uf.find(u)
        edge_count[root] += 1
    
    result = 0
    for r in size:
        k = size[r]
        m_i = edge_count.get(r, 0)
        result += (k * (k - 1) // 2) - m_i
    
    print(result)

if __name__ == '__main__':
    main()