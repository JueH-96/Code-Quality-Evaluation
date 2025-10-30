# Precompute mex table
mex_table = [[[0] * 3 for _ in range(3)] for __ in range(3)]
for m in range(3):
    for e in range(3):
        for x in range(3):
            s = {m, e, x}
            mex = 0
            while mex in s:
                mex += 1
            mex_table[m][e][x] = mex

# Read input
N = int(input())
A = list(map(int, input().split()))
S = input().strip()

# Compute prefix_m: counts of M's before each position
prefix_m = [[0, 0, 0]] * (N + 1)
current = [0, 0, 0]
prefix_m[0] = current.copy()
for i in range(N):
    if S[i] == 'M':
        a = A[i]
        current[a] += 1
    prefix_m[i + 1] = current.copy()

# Compute suffix_x: counts of X's after each position
suffix_x = [[0, 0, 0]] * (N + 1)
current = [0, 0, 0]
suffix_x[N] = current.copy()
for i in range(N - 1, -1, -1):
    if S[i] == 'X':
        a = A[i]
        current[a] += 1
    suffix_x[i] = current.copy()

# Calculate the total sum
total = 0
for j in range(N):
    if S[j] == 'E':
        m_counts = prefix_m[j]
        x_counts = suffix_x[j + 1]
        e_val = A[j]
        for m in range(3):
            for x in range(3):
                total += m_counts[m] * x_counts[x] * mex_table[m][e_val][x]

print(total)