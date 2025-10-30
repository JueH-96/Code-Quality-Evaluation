import sys

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
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    operations = []
    for _ in range(M):
        K_i = int(input[ptr])
        ptr += 1
        C_i = int(input[ptr])
        ptr += 1
        S_i = list(map(int, input[ptr:ptr+K_i]))
        ptr += K_i
        operations.append((C_i, S_i))
    # Sort operations by C_i ascending
    operations.sort()
    dsu = DSU(N)
    total = 0
    for C_i, S_i in operations:
        roots = set()
        for node in S_i:
            root = dsu.find(node)
            roots.add(root)
        k = len(roots)
        total += (k - 1) * C_i
        if k > 1:
            roots_list = list(roots)
            main_root = roots_list[0]
            for r in roots_list[1:]:
                dsu.union(main_root, r)
    # Check if all nodes are connected
    root = dsu.find(1)
    if dsu.size[root] == N:
        print(total)
    else:
        print(-1)

if __name__ == '__main__':
    main()