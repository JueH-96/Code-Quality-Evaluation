import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    edges = []
    
    for _ in range(M):
        K_i = int(input[ptr])
        ptr += 1
        C_i = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+K_i]))
        ptr += K_i
        rep = A[0]
        for node in A[1:]:
            edges.append((C_i, rep, node))
    
    edges.sort()
    
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size + 1))
            self.rank = [1] * (size + 1)
            self.count = size  # Number of components
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            xroot = self.find(x)
            yroot = self.find(y)
            if xroot == yroot:
                return False
            if self.rank[xroot] < self.rank[yroot]:
                self.parent[xroot] = yroot
            else:
                self.parent[yroot] = xroot
                if self.rank[xroot] == self.rank[yroot]:
                    self.rank[xroot] += 1
            self.count -= 1
            return True
    
    dsu = DSU(N)
    total = 0
    for e in edges:
        c, u, v = e
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            total += c
    
    print(total if dsu.count == 1 else -1)

if __name__ == "__main__":
    main()