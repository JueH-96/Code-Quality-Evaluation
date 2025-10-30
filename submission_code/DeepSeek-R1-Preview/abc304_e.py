import sys
from sys import stdin

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
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr +=1
    uf = UnionFind(N)
    for _ in range(M):
        u = int(input[ptr])
        ptr +=1
        v = int(input[ptr])
        ptr +=1
        uf.union(u, v)
    K = int(input[ptr])
    ptr +=1
    forbidden = set()
    for _ in range(K):
        x = int(input[ptr])
        ptr +=1
        y = int(input[ptr])
        ptr +=1
        rx = uf.find(x)
        ry = uf.find(y)
        a, b = sorted([rx, ry])
        forbidden.add( (a, b) )
    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        p = int(input[ptr])
        ptr +=1
        q = int(input[ptr])
        ptr +=1
        rp = uf.find(p)
        rq = uf.find(q)
        if rp == rq:
            print("Yes")
        else:
            a, b = sorted([rp, rq])
            if (a, b) in forbidden:
                print("No")
            else:
                print("Yes")

if __name__ == "__main__":
    main()