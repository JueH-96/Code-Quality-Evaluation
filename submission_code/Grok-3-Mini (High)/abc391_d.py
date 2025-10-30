import sys
data = sys.stdin.read().split()
data = [int(x) for x in data]
index = 0
N = data[index]
W = data[index + 1]
index += 2

# Read blocks
block_data = []
for i in range(N):
    X = data[index]
    Y = data[index + 1]
    index += 2
    block_data.append((X, Y))

# Group blocks by column with block index
cols = [[] for _ in range(W)]  # cols[col_idx] for column 1 to W, index 0 to W-1
for idx in range(N):  # idx is block index 0 to N-1
    X, Y = block_data[idx]
    col_idx = X - 1  # 0-based column index
    cols[col_idx].append((Y, idx))

# Sort each column by Y
for col_idx in range(W):
    cols[col_idx].sort()  # sort by Y, first element of tuple

# Assign ranks to each block (1-based rank in their column)
rank_of_block = [0] * N
for col_idx in range(W):
    sorted_col = cols[col_idx]
    for pos in range(len(sorted_col)):  # pos 0-based
        Y, block_idx = sorted_col[pos]
        rank_of_block[block_idx] = pos + 1  # 1-based rank

# Compute M (min number of blocks per column) and max_len
num_blocks_per_col = [len(col) for col in cols]
M = min(num_blocks_per_col)
max_len = max(num_blocks_per_col)

# Compute max_y_R for each rank R (max Y over all blocks with that rank)
max_y_R = [0] * (max_len + 1)  # index 0 unused, 1 to max_len, init to 0
for idx in range(N):
    R = rank_of_block[idx]  # 1-based rank
    Y = block_data[idx][1]  # Y coordinate
    if max_y_R[R] < Y:
        max_y_R[R] = Y

# Read Q
Q = data[index]
index += 1

# Read queries
queries = []
for q in range(Q):
    T_j = data[index]
    A_j = data[index + 1]  # A_j is 1-based block index
    index += 2
    queries.append((T_j, A_j))

# Process each query and output
for T_j, A_j in queries:
    idx = A_j - 1  # 0-based block index
    R = rank_of_block[idx]  # 1-based rank
    if R > M or max_y_R[R] > T_j:
        print("Yes")
    else:
        print("No")