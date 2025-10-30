import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))
        self.rank = [1] * (n+1)
    
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
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    
    dsu = DSU(N)
    for _ in range(M):
        u = int(data[ptr])
        ptr += 1
        v = int(data[ptr])
        ptr += 1
        dsu.union(u, v)
    
    K = int(data[ptr])
    ptr += 1
    forbidden = set()
    for _ in range(K):
        x = int(data[ptr])
        ptr += 1
        y = int(data[ptr])
        ptr += 1
        rx = dsu.find(x)
        ry = dsu.find(y)
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))
    
    Q = int(data[ptr])
    ptr += 1
    output = []
    for _ in range(Q):
        p = int(data[ptr])
        ptr += 1
        q = int(data[ptr])
        ptr += 1
        rp = dsu.find(p)
        rq = dsu.find(q)
        if rp == rq:
            output.append("Yes")
        else:
            a, b = sorted([rp, rq])
            if (a, b) in forbidden:
                output.append("No")
            else:
                output.append("Yes")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()