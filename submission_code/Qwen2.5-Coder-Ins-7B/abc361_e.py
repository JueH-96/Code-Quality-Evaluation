import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []

for i in range(N-1):
    A = int(data[3*i+1])
    B = int(data[3*i+2])
    C = int(data[3*i+3])
    edges.append((A, B, C))

# Function to find the minimum spanning tree using Kruskal's algorithm
def kruskal_mst():
    edges.sort(key=lambda x: x[2])
    parent = list(range(N+1))
    rank = [0] * (N+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    mst_cost = 0
    for A, B, C in edges:
        if find(A) != find(B):
            union(A, B)
            mst_cost += C

    return mst_cost

# The minimum travel distance required to visit all cities at least once is the minimum spanning tree cost
print(kruskal_mst())