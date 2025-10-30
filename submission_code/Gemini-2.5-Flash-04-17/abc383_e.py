import sys

# Set recursion depth for path compression in DSU if needed, though usually not necessary with Python's default limit
# sys.setrecursionlimit(200005)

# DSU implementation
class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        # cnt_a[i]: number of A vertices in component rooted at i
        self.cnt_a = [0] * (n + 1)
        # cnt_b[i]: number of B vertices in component rooted at i
        self.cnt_b = [0] * (n + 1)
        # size[i]: size of the component rooted at i (number of vertices)
        self.size = [1] * (n + 1)

    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, u, v, w):
        root_u = self.find(u)
        root_v = self.find(v)
        total_cost = 0

        if root_u != root_v:
            # Union by size optimization
            if self.size[root_u] < self.size[root_v]:
                root_u, root_v = root_v, root_u # Swap to merge smaller into larger

            # Calculate contribution before merging counts
            k_u = self.cnt_a[root_u]
            l_u = self.cnt_b[root_u]
            k_v = self.cnt_a[root_v]
            l_v = self.cnt_b[root_v]

            # Number of pairs matchable within root_u using edges strictly < w
            # This is implicitly captured by the counts k_u, l_u which were formed by previous unions
            matchable_u_before = min(k_u, l_u)
            # Number of pairs matchable within root_v using edges strictly < w
            matchable_v_before = min(k_v, l_v)
            
            # Number of pairs matchable within root_u U root_v using edges up to weight w
            matchable_combined_after = min(k_u + k_v, l_u + l_v)

            # Number of newly matchable pairs that must cross the boundary
            # These pairs have minimum bottleneck weight exactly w
            newly_matchable = matchable_combined_after - (matchable_u_before + matchable_v_before)

            total_cost += newly_matchable * w

            # Perform union
            self.parent[root_v] = root_u
            self.size[root_u] += self.size[root_v]
            self.cnt_a[root_u] += self.cnt_a[root_v]
            self.cnt_b[root_u] += self.cnt_b[root_v]

        return total_cost

# Read input
N, M, K = map(int, sys.stdin.readline().split())

edges = []
for _ in range(M):
    u, v, w = map(int, sys.stdin.readline().split())
    edges.append((u, v, w))

A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Sort edges by weight
edges.sort(key=lambda item: item[2])

# Initialize DSU
dsu = DSU(N)

# Populate initial counts for A and B based on their initial components (single vertices)
# We find the root for each vertex, which is itself initially, and increment counts there.
# This correctly sets up the counts for each initial component.
for vertex_a in A:
    dsu.cnt_a[dsu.find(vertex_a)] += 1

for vertex_b in B:
    dsu.cnt_b[dsu.find(vertex_b)] += 1

# Process edges in increasing order of weight and calculate minimum sum
min_total_sum = 0

for u, v, w in edges:
    min_total_sum += dsu.union(u, v, w)

# Print the result
print(min_total_sum)