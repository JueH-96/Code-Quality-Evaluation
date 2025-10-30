import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

# Initialize the cells and their colors
cells = list(range(1, N + 1))

# Union-Find structure to manage connected components
parent = list(range(N + 1))
rank = [0] * (N + 1)

def find(u):
    if parent[u] != u:
        parent[u] = find(parent[u])
    return parent[u]

def union(u, v):
    root_u = find(u)
    root_v = find(v)
    if root_u != root_v:
        if rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_u] = root_v
            if rank[root_u] == rank[root_v]:
                rank[root_v] += 1

# Initialize the color count
color_count = [1] * (N + 1)

# Process the queries
index = 2
results = []
for _ in range(Q):
    query_type = int(data[index])
    if query_type == 1:
        x = int(data[index + 1])
        c = int(data[index + 2])
        original_color = cells[x - 1]
        if original_color != c:
            # Find all cells in the same component
            root = find(x)
            for i in range(1, N + 1):
                if find(i) == root and cells[i - 1] == original_color:
                    cells[i - 1] = c
            color_count[original_color] = 0
        color_count[c] = cells.count(c)
        index += 3
    elif query_type == 2:
        c = int(data[index + 1])
        results.append(color_count[c])
        index += 2

# Output the results
sys.stdout.write("
".join(map(str, results)) + "
")