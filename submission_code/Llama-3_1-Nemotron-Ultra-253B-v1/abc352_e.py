import sys

class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))  # 1-based indexing
        self.rank = [1] * (size + 1)
    
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
        A = list(map(int, input[ptr:ptr+K_i]))
        ptr += K_i
        operations.append((C_i, K_i, A))
    
    operations.sort()
    dsu = DSU(N)
    total = 0
    
    for C_i, K_i, A in operations:
        roots = set()
        for u in A:
            roots.add(dsu.find(u))
        count = len(roots)
        if count >= 1:
            total += (count - 1) * C_i
            if count > 1:
                main = roots.pop()
                for r in roots:
                    dsu.union(main, r)
    
    root = dsu.find(1)
    for i in range(2, N + 1):
        if dsu.find(i) != root:
            print(-1)
            return
    print(total)

if __name__ == '__main__':
    main()