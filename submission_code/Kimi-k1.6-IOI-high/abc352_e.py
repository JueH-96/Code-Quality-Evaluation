import sys

class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.size = [1] * (size + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]
        return True

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    ops = []
    for _ in range(M):
        K_i = int(input[ptr])
        ptr += 1
        C_i = int(input[ptr])
        ptr += 1
        S_i = list(map(int, input[ptr:ptr + K_i]))
        ptr += K_i
        ops.append((C_i, S_i))
    
    ops.sort()
    
    dsu = DSU(N)
    total = 0
    
    for C_i, S_i in ops:
        roots = set()
        for node in S_i:
            roots.add(dsu.find(node))
        m = len(roots)
        if m >= 1:
            total += (m - 1) * C_i
            if m > 1:
                roots_list = list(roots)
                main = roots_list[0]
                for r in roots_list[1:]:
                    dsu.union(main, r)
    
    root = dsu.find(1)
    if dsu.size[root] == N:
        print(total)
    else:
        print(-1)

if __name__ == '__main__':
    main()