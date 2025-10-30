import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))
        self.size = [1]*(n+1)
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
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
    dsu = DSU(N)
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        dsu.union(u, v)
    K = int(input[ptr])
    ptr += 1
    forbidden = set()
    for _ in range(K):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        rx = dsu.find(x)
        ry = dsu.find(y)
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))
    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        p = int(input[ptr])
        ptr += 1
        q = int(input[ptr])
        ptr += 1
        rp = dsu.find(p)
        rq = dsu.find(q)
        if rp == rq:
            print("Yes")
        else:
            a, b = sorted((rp, rq))
            if (a, b) in forbidden:
                print("No")
            else:
                print("Yes")

if __name__ == "__main__":
    main()