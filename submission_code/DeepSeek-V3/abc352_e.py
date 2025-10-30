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
        # Add edges for all pairs in S with weight C
        # To optimize, we can add edges between the first element and the rest
        # This way, we only add K-1 edges instead of K*(K-1)/2
        first = S[0]
        for u in S[1:]:
            edges.append((C, first, u))
    # Sort edges by weight
    edges.sort()
    uf = UnionFind(N)
    total_weight = 0
    for weight, u, v in edges:
        if uf.union(u, v):
            total_weight += weight
    # Check if all nodes are connected
    root = uf.find(1)
    for i in range(2, N+1):
        if uf.find(i) != root:
            print(-1)
            return
    print(total_weight)

if __name__ == "__main__":
    main()