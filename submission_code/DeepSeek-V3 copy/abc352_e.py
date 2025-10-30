import sys
import heapq

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.size = [1] * (size + 1)
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.size[xroot] < self.size[yroot]:
            xroot, yroot = yroot, xroot
        self.parent[yroot] = xroot
        self.size[xroot] += self.size[yroot]
        return True

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    edges = []
    for _ in range(M):
        K = int(data[idx])
        C = int(data[idx+1])
        idx += 2
        S = list(map(int, data[idx:idx+K]))
        idx += K
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                u = S[i]
                v = S[j]
                if u < v:
                    edges.append((C, u, v))
    uf = UnionFind(N)
    edges.sort()
    total_weight = 0
    for cost, u, v in edges:
        if uf.union(u, v):
            total_weight += cost
    root = uf.find(1)
    for i in range(2, N+1):
        if uf.find(i) != root:
            print(-1)
            return
    print(total_weight)

if __name__ == "__main__":
    main()