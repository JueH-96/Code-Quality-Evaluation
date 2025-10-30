import sys

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.rank = [0] * (size + 1)
        self.count = size

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
        self.count -= 1

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
    operations.sort()
    uf = UnionFind(N)
    total = 0
    for C, S in operations:
        roots = set()
        unique_roots = []
        for node in S:
            r = uf.find(node)
            if r not in roots:
                roots.add(r)
                unique_roots.append(r)
        k = len(unique_roots)
        total += (k - 1) * C
        if k > 0:
            first = unique_roots[0]
            for r in unique_roots[1:]:
                uf.union(first, r)
    print(total if uf.count == 1 else -1)

if __name__ == "__main__":
    main()