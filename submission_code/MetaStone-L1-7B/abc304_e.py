import sys
sys.setrecursionlimit(1 << 25)

class DSU:
    def __init__(self, size):
        self.parent = list(range(size + 1))
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

    dsu = DSU(N)
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        dsu.union(u, v)
    
    K = int(input[ptr])
    ptr += 1
    S = set()
    for _ in range(K):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        rx = dsu.find(x)
        ry = dsu.find(y)
        S.add((rx, ry))
    
    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        p = int(input[ptr])
        ptr += 1
        q = int(input[ptr])
        ptr += 1
        rp = dsu.find(p)
        rq = dsu.find(q)
        if (rp, rq) in S or (rq, rp) in S:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()