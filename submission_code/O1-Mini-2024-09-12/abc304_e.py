# YOUR CODE HERE
import sys

class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx == fy:
            return
        self.parent[fy] = fx

def main():
    import sys
    import sys
    import sys
    input = sys.stdin.read().split()
    it = iter(input)
    
    N = int(next(it))
    M = int(next(it))
    
    dsu = DSU(N)
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        dsu.union(u, v)
    
    K = int(next(it))
    forbidden = set()
    for _ in range(K):
        x = int(next(it))
        y = int(next(it))
        fx = dsu.find(x)
        fy = dsu.find(y)
        if fx > fy:
            fx, fy = fy, fx
        forbidden.add((fx, fy))
    
    Q = int(next(it))
    res = []
    for _ in range(Q):
        p = int(next(it))
        q = int(next(it))
        fp = dsu.find(p)
        fq = dsu.find(q)
        if fp == fq:
            res.append("Yes")
        else:
            a, b = (fp, fq) if fp < fq else (fq, fp)
            if (a, b) in forbidden:
                res.append("No")
            else:
                res.append("Yes")
    
    print('
'.join(res))

if __name__ == "__main__":
    main()