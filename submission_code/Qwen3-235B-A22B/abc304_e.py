import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))  # 1-based indexing
        self.rank = [1] * (n + 1)
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        # Union by rank
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
    
    forbidden_pairs = set()
    K = int(input[ptr])
    ptr += 1
    for _ in range(K):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        rx = dsu.find(x)
        ry = dsu.find(y)
        if rx < ry:
            forbidden_pairs.add((rx, ry))
        else:
            forbidden_pairs.add((ry, rx))
    
    Q = int(input[ptr])
    ptr += 1
    output = []
    for _ in range(Q):
        p = int(input[ptr])
        ptr += 1
        q = int(input[ptr])
        ptr += 1
        rp = dsu.find(p)
        rq = dsu.find(q)
        if rp == rq:
            output.append("Yes")
        else:
            if rp < rq:
                pair = (rp, rq)
            else:
                pair = (rq, rp)
            if pair in forbidden_pairs:
                output.append("No")
            else:
                output.append("Yes")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()