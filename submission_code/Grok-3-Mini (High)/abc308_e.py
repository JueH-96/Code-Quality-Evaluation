import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
S = data[index]

# Define mex_three function
def mex_three(x, y, z):
    has_zero = (x == 0 or y == 0 or z == 0)
    has_one = (x == 1 or y == 1 or z == 1)
    has_two = (x == 2 or y == 2 or z == 2)
    if not has_zero:
        return 0
    elif not has_one:
        return 1
    elif not has_two:
        return 2
    else:
        return 3

# Compute cumulative sums for M
cum_M = [[0 for _ in range(N + 1)] for _ in range(3)]
for val in range(3):
    for p in range(1, N + 1):
        if S[p - 1] == 'M' and A[p - 1] == val:
            cum_M[val][p] = cum_M[val][p - 1] + 1
        else:
            cum_M[val][p] = cum_M[val][p - 1]

# Compute suffix sums for X
suf_X = [[0 for _ in range(N + 1)] for _ in range(3)]
for val in range(3):
    for p in range(N - 1, -1, -1):
        contrib = 1 if S[p] == 'X' and A[p] == val else 0
        suf_X[val][p] = contrib + suf_X[val][p + 1]

# Sum over all j where S[j] == 'E'
total_sum = 0
for j in range(N):
    if S[j] == 'E':
        v = A[j]
        for a in range(3):
            left_cnt = cum_M[a][j]
            for b in range(3):
                right_cnt = suf_X[b][j + 1]
                mex_val = mex_three(a, v, b)
                total_sum += left_cnt * right_cnt * mex_val

# Output the result
print(total_sum)