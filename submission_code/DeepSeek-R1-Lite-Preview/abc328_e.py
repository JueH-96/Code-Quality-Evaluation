import sys
from itertools import combinations

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1
        return True

def main():
    input = sys.stdin.read().splitlines()
    N, M, K = map(int, input[0].split())
    edges = []
    for i in range(1, M + 1):
        u, v, w = map(int, input[i].split())
        u -= 1
        v -= 1
        edges.append((u, v, w))
    min_sum = K
    for combo in combinations(edges, N - 1):
        dsu = DSU(N)
        sum_weights = 0
        for edge in combo:
            u, v, w = edge
            if dsu.union(u, v):
                sum_weights += w
            else:
                break
        else:
            root = dsu.find(0)
            if all(dsu.find(i) == root for i in range(N)):
                current_sum_mod = sum_weights % K
                if current_sum_mod < min_sum:
                    min_sum = current_sum_mod
    print(min_sum)

if __name__ == '__main__':
    main()