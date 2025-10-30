import sys
from sys import stdin
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, stdin.readline().split())
        edges.append((w, u-1, v-1))  # 0-based
    
    A = list(map(lambda x: int(x)-1, stdin.readline().split()))
    B = list(map(lambda x: int(x)-1, stdin.readline().split()))
    
    # Compute a_count and b_count
    a_count = defaultdict(int)
    for x in A:
        a_count[x] += 1
    b_count = defaultdict(int)
    for x in B:
        b_count[x] += 1
    
    # Compute MST using Kruskal's algorithm
    parent = list(range(N))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[v_root] = u_root
        return True
    
    # Sort edges in increasing order of weight
    edges.sort()
    mst_edges = []
    for w, u, v in edges:
        if union(u, v):
            mst_edges.append((w, u, v))
        if len(mst_edges) == N-1:
            break
    
    # Re-initialize Union-Find with a and b counts
    class UnionFind:
        def __init__(self, a, b):
            self.parent = list(range(N))
            self.a = a.copy()
            self.b = b.copy()
        
        def find(self, u):
            if self.parent[u] != u:
                self.parent[u] = self.find(self.parent[u])
            return self.parent[u]
        
        def union(self, u, v, w, remaining, sum_total):
            u_root = self.find(u)
            v_root = self.find(v)
            if u_root == v_root:
                return remaining, sum_total
            a1, b1 = self.a[u_root], self.b[u_root]
            a2, b2 = self.a[v_root], self.b[v_root]
            new_pairs = min(a1, b2) + min(a2, b1)
            take = min(new_pairs, remaining)
            sum_total += w * take
            remaining -= take
            # Update a and b for the merged component
            merged_a = (a1 - min(a1, b2)) + (a2 - min(a2, b1))
            merged_b = (b2 - min(a1, b2)) + (b1 - min(a2, b1))
            # Merge into u_root
            self.parent[v_root] = u_root
            self.a[u_root] = merged_a
            self.b[u_root] = merged_b
            return remaining, sum_total
    
    # Initialize a and b arrays
    a = [0] * N
    for x in a_count:
        a[x] = a_count[x]
    b = [0] * N
    for x in b_count:
        b[x] = b_count[x]
    
    uf = UnionFind(a, b)
    sum_total = 0
    remaining = K
    
    # Process MST edges in increasing order
    for w, u, v in mst_edges:
        remaining, sum_total = uf.union(u, v, w, remaining, sum_total)
        if remaining == 0:
            break
    
    print(sum_total)

if __name__ == '__main__':
    main()