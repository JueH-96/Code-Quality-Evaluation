import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))  # 1-based indexing
        self.size = [1] * (n+1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        # Union by size
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    dsu = DSU(N)
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        dsu.union(a, b)
    
    # Collect component sizes
    component_sizes = {}
    for i in range(1, N+1):
        root = dsu.find(i)
        if root not in component_sizes:
            component_sizes[root] = dsu.size[root]
    
    sum_edges = 0
    for s in component_sizes.values():
        sum_edges += s * (s - 1) // 2
    
    print(sum_edges - M)

if __name__ == '__main__':
    main()