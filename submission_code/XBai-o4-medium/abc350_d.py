class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))  # 1-based indexing
        self.size = [1] * (n + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    dsu = DSU(N)
    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        dsu.union(a, b)
    
    component_sizes = []
    for i in range(1, N + 1):
        if dsu.find(i) == i:
            component_sizes.append(dsu.size[i])
    
    total = 0
    for s in component_sizes:
        total += s * (s - 1) // 2
    
    ans = total - M
    print(ans)

if __name__ == '__main__':
    main()