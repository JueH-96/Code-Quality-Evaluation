import sys
from sys import stdin

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    K = int(data[idx])
    idx +=1
    
    edges = []
    for _ in range(M):
        u = int(data[idx])
        idx +=1
        v = int(data[idx])
        idx +=1
        w = int(data[idx])
        idx +=1
        edges.append((w, u, v))
    
    A = list(map(int, data[idx:idx+K]))
    idx +=K
    B = list(map(int, data[idx:idx+K]))
    idx +=K
    
    count_A = [0]*(N+1)
    for a in A:
        count_A[a] +=1
    
    count_B = [0]*(N+1)
    for b in B:
        count_B[b] +=1
    
    edges.sort()
    
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size+1))
            self.rank = [0]*(size+1)
            self.count_A = [0]*(size+1)
            self.count_B = [0]*(size+1)
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
    
    dsu = DSU(N)
    for u in range(1, N+1):
        dsu.count_A[u] = count_A[u]
        dsu.count_B[u] = count_B[u]
    
    sum_total = 0
    for w, u, v in edges:
        ru = dsu.find(u)
        rv = dsu.find(v)
        if ru != rv:
            a = dsu.count_A[ru]
            b = dsu.count_B[ru]
            c = dsu.count_A[rv]
            d = dsu.count_B[rv]
            pairs = min(a, d) + min(b, c)
            sum_total += pairs * w
            if dsu.rank[ru] > dsu.rank[rv]:
                dsu.parent[rv] = ru
                dsu.count_A[ru] = a + c - pairs
                dsu.count_B[ru] = b + d - pairs
            else:
                dsu.parent[ru] = rv
                dsu.count_A[rv] = a + c - pairs
                dsu.count_B[rv] = b + d - pairs
                if dsu.rank[ru] == dsu.rank[rv]:
                    dsu.rank[rv] +=1
    print(sum_total)

if __name__ == "__main__":
    main()