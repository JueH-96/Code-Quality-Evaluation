from typing import List

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.count = size

    def find(self, p):
        if p != self.parent[p]:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        elif self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1
        self.count -= 1

def answer(N: int, M: int, operations: List[List[int]]) -> int:
    uf = UnionFind(N+1)  # 1-indexed
    edges = []
    node_group = [0] * (N+1)

    for operation in operations:
        K, C = operation[0], operation[1]
        nodes = operation[2:]
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if uf.find(nodes[i]) != uf.find(nodes[j]):
                    uf.union(nodes[i], nodes[j])
                    # Record the cost for a potential path from one component to another
                    if node_group[nodes[i]] == 0:
                        node_group[nodes[i]] = C
                    if node_group[nodes[j]] == 0:
                        node_group[nodes[j]] = C
                    edges.append((C, nodes[i], nodes[j]))
        
    # Add edges within the groups if not already connected
    for i in range(1, N+1):
        if node_group[i] != 0:
            for j in range(i+1, N+1):
                if uf.find(i) != uf.find(j) and node_group[j] == 0:
                    node_group[j] = node_group[i]
                    edges.append((node_group[i], i, j))

    # Sort the edges by weight
    edges.sort()

    mst_cost = 0
    for weight, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_cost += weight
    
    if uf.count > 1:
        return -1
    else:
        return mst_cost

if __name__ == "__main__":
    N, M = map(int, input().split())
    operations = []
    for _ in range(M):
        line1 = list(map(int, input().split()))
        line2 = list(map(int, input().split()))
        operations.append(line1 + line2)
    print(answer(N, M, operations))