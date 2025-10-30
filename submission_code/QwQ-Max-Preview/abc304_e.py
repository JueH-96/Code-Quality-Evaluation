import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))  # 1-based indexing
        self.rank = [1] * (n + 1)
    
    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]  # Path compression
            u = self.parent[u]
        return u
    
    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
        # Union by rank
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        else:
            self.parent[v_root] = u_root
            if self.rank[u_root] == self.rank[v_root]:
                self.rank[u_root] += 1

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
        cx = dsu.find(x)
        cy = dsu.find(y)
        if cx > cy:
            cx, cy = cy, cx
        forbidden.add((cx, cy))
    
    Q = int(input[ptr])
    ptr += 1
    output = []
    for _ in range(Q):
        p = int(input[ptr])
        ptr += 1
        q = int(input[ptr])
        ptr += 1
        cp = dsu.find(p)
        cq = dsu.find(q)
        if cp == cq:
            output.append("Yes")
        else:
            if cp > cq:
                cp, cq = cq, cp
            if (cp, cq) in forbidden:
                output.append("No")
            else:
                output.append("Yes")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()