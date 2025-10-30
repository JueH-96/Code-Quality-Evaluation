class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.component_weight = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y, weight_x, weight_y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
                self.component_weight[rootX] += self.component_weight[rootY]
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
                self.component_weight[rootY] += self.component_weight[rootX]
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
                self.component_weight[rootX] += self.component_weight[rootY]

    def add_weight(self, x, weight):
        rootX = self.find(x)
        self.component_weight[rootX] += weight

    def get_component_weight(self, x):
        rootX = self.find(x)
        return self.component_weight[rootX]

import sys
input = sys.stdin.read
def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    weights = list(map(int, data[index:index + N]))
    index += N
    
    intervals = []
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        intervals.append((L, R))
        index += 2
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        s = int(data[index]) - 1
        t = int(data[index + 1]) - 1
        queries.append((s, t))
        index += 2
    
    dsu = DSU(N)
    
    for i in range(N):
        dsu.add_weight(i, weights[i])
    
    for i in range(N):
        L_i, R_i = intervals[i]
        for j in range(i + 1, N):
            L_j, R_j = intervals[j]
            if R_i < L_j or R_j < L_i:
                dsu.union(i, j, weights[i], weights[j])
    
    results = []
    for s, t in queries:
        if dsu.find(s) == dsu.find(t):
            results.append(dsu.get_component_weight(s))
        else:
            results.append(-1)
    
    for result in results:
        print(result)