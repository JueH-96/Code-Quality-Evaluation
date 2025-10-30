import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
grid = []
for _ in range(N):
    row_str = data[index]
    index += 1
    row = [1 if c == 'B' else 0 for c in row_str]
    grid.append(row)
# compute cum_black[k][y]: sum x=0 to k grid[x][y]
cum_black = [[0 for y in range(N)] for k in range(N)]
for y in range(N):
    cum_sum = 0
    for k in range(N):
        cum_sum += grid[k][y]
        cum_black[k][y] = cum_sum
# compute col_black_sum[y]
col_black_sum = [cum_black[N-1][y] for y in range(N)]
# compute cum_col_sum[s]
cum_col_sum = [0]
sum_so_far = 0
for val in col_black_sum:
    sum_so_far += val
    cum_col_sum.append(sum_so_far)
# compute prefix_C for each k
prefix_C = []
for k in range(N):
    prefix_sum_k = [0]
    current_sum = 0
    for y in range(N):
        current_sum += cum_black[k][y]
        prefix_sum_k.append(current_sum)
    prefix_C.append(prefix_sum_k)
# define compute_S function
def compute_S(i, j):
    if i < 0 or j < 0:
        return 0
    i_div = i // N
    i_mod = i % N
    j_div = j // N
    j_mod = j % N
    sum1 = (j_div + 1) * cum_col_sum[j_mod + 1] + j_div * (cum_col_sum[N] - cum_col_sum[j_mod + 1])
    k_val = i_mod
    sum2 = (j_div + 1) * prefix_C[k_val][j_mod + 1] + j_div * (prefix_C[k_val][N] - prefix_C[k_val][j_mod + 1])
    return (i_div * sum1 + sum2)
# now process Q queries
for _ in range(Q):
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    D = int(data[index])
    index += 1
    ans = compute_S(C, D) - compute_S(A - 1, D) - compute_S(C, B - 1) + compute_S(A - 1, B - 1)
    print(ans)