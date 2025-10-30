import sys
import bisect
from collections import defaultdict

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
Y = int(data[index])
index += 1
A = []
for i in range(H):
    row = [int(x) for x in data[index:index + W]]
    A.append(row)
    index += W

total_cells = H * W

# Group cells by elevation
group_dict = defaultdict(list)
for i in range(H):
    for j in range(W):
        a_val = A[i][j]
        idx = i * W + j
        group_dict[a_val].append(idx)

elevations = sorted(group_dict.keys())

num_cells_total = H * W
parent = [-1] * num_cells_total
size_list = [0] * num_cells_total
bc_list = [False] * num_cells_total
sunk_size = 0

# Define find function
def find(idx):
    global parent
    root = idx
    while parent[root] != root:
        root = parent[root]
    # Path compression
    current = idx
    while current != root:
        next_p = parent[current]
        parent[current] = root
        current = next_p
    return root

# Define union function
def union(a, b):
    global parent, size_list, bc_list, sunk_size
    rootA = find(a)
    rootB = find(b)
    if rootA == rootB:
        return
    sizeA = size_list[rootA]
    sizeB = size_list[rootB]
    bcA = bc_list[rootA]
    bcB = bc_list[rootB]
    if bcA != bcB:
        if not bcA:
            sunk_size += sizeA
        else:
            sunk_size += sizeB
    # Union them
    if sizeA < sizeB:
        parent[rootA] = rootB
        size_list[rootB] += sizeA
        bc_list[rootB] = bcA or bcB
    else:
        parent[rootB] = rootA
        size_list[rootA] += sizeB
        bc_list[rootA] = bcA or bcB

# Sunk size changes
sunk_changes = []

for E in elevations:
    new_cells = group_dict[E]
    new_set = set(new_cells)
    # Add each new cell as singleton
    for idx in new_cells:
        i = idx // W
        j = idx % W
        is_bound = (i == 0 or i == H - 1 or j == 0 or j == W - 1)
        parent[idx] = idx
        size_list[idx] = 1
        bc_list[idx] = is_bound
        if is_bound:
            sunk_size += 1
    # Union among adjacent new cells (right and down to avoid duplicates)
    for idx in new_cells:
        i = idx // W
        j = idx % W
        # Right neighbor
        if j < W - 1:
            neigh_idx = idx + 1
            if neigh_idx in new_set:
                union(idx, neigh_idx)
        # Down neighbor
        if i < H - 1:
            neigh_idx_down = idx + W
            if neigh_idx_down in new_set:
                union(idx, neigh_idx_down)
    # Union with adjacent old cells (A < E)
    for idx in new_cells:
        i = idx // W
        j = idx % W
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neigh_idx = ni * W + nj
                if A[ni][nj] < E:  # Old cell
                    union(idx, neigh_idx)
    # Record sunk size after adding elevation E
    sunk_changes.append((E, sunk_size))

# Extract E values and sunk size values for bisect
if sunk_changes:
    E_values = [pair[0] for pair in sunk_changes]
    sunk_size_values = [pair[1] for pair in sunk_changes]
else:
    E_values = []
    sunk_size_values = []

# Output for each year from 1 to Y
for year in range(1, Y + 1):
    S_query = year
    # Find the index of the largest E <= S_query
    idx = bisect.bisect_right(E_values, S_query) - 1
    if idx >= 0 and idx < len(E_values):
        sunk_sz = sunk_size_values[idx]
    else:
        sunk_sz = 0
    remaining = total_cells - sunk_sz
    print(remaining)