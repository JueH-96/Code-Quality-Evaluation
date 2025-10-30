import sys
sys.setrecursionlimit(200000)
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
out_adj = [[] for _ in range(N+1)]  # list of (B, C_test) for each A
for i in range(M):
    A = int(data[index])
    B = int(data[index+1])
    C = int(data[index+2])  # c_test
    index += 3
    out_adj[A].append((B, C))

# Create added_edges for H constraints
added_edges = []
 for A in range(1, N+1):
     neighbors = out_adj[A]
     if len(neighbors) > 1:
         B0, c0 = neighbors[0]
         for B, c in neighbors[1:]:
             w = c ^ c0
             added_edges.append((B, B0, w))  # add (B, B0, w)

# Union-Find with weights
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size
        self.dist = [0] * size

    def find(self, u):
        if self.parent[u] != u:
            root, d = self.find(self.parent[u])
            self.dist[u] = self.dist[u] ^ d
            self.parent[u] = root
            return root, self.dist[u]
        else:
            return u, 0

    def union(self, u, v, w):
        root_u, d_u = self.find(u)
        root_v, d_v = self.find(v)
        if root_u == root_v:
            if (d_u ^ d_v) != w:
                return False
            else:
                return True
        else:
            if self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                self.dist[root_u] = w ^ d_u ^ d_v
            elif self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                self.dist[root_v] = w ^ d_u ^ d_v
            else:
                self.parent[root_u] = root_v
                self.dist[root_u] = w ^ d_u ^ d_v
                self.rank[root_v] += 1
            return True

uf = UnionFind(N+1)

# Add all added edges
for edge in added_edges:
    u, v, w = edge
    if not uf.union(u, v, w):
        print(-1)
        sys.exit()

# No conflict, apply path compression for all nodes
for u in range(1, N+1):
    uf.find(u)

# Now dist[u] is H_u with H_root=0
# Compute C for each A
C_values = []
for A in range(1, N+1):
    if out_adj[A]:  # out-degree >0
        B, c = out_adj[A][0]  # first one
        H_a = uf.dist[A]
        H_b = uf.dist[B]
        C_a = H_a ^ H_b ^ c
        C_values.append(C_a)
    else:
        C_a = 0
        C_values.append(C_a)

# Output the string
print(''.join(map(str, C_values)))