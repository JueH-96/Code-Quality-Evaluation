class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))  # 1-based indexing
        self.size = [1] * (n + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        # Union by size: attach smaller tree to larger tree
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    m = int(input[idx])
    idx += 1
    if m == 0:
        print(0)
        return
    dsu = DSU(n)
    for _ in range(m):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        dsu.union(a, b)
    total = 0
    # Calculate the sum of (k choose 2) for each component
    for i in range(1, n + 1):
        if dsu.find(i) == i:
            k = dsu.size[i]
            total += k * (k - 1) // 2
    # Subtract the initial number of edges
    print(total - m)

if __name__ == "__main__":
    main()