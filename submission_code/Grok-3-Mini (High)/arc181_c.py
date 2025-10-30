import sys
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
P_list = data[index:index + N]
index += N
Q_list = data[index:index + N]
# Create row_rank
row_rank = [0] * (N + 1)
for rank in range(1, N + 1):
    row_idx = P_list[rank - 1]
    row_rank[row_idx] = rank
# Create col_rank
col_rank = [0] * (N + 1)
for rank in range(1, N + 1):
    col_idx = Q_list[rank - 1]
    col_rank[col_idx] = rank
# Output the grid
for i in range(1, N + 1):
    row_str = ''
    for j in range(1, N + 1):
        if row_rank[i] + col_rank[j] > N:
            row_str += '1'
        else:
            row_str += '0'
    print(row_str)