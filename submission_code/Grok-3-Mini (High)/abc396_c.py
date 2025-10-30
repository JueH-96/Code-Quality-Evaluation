import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
B = list(map(int, data[index:index + N]))
index += N
W = list(map(int, data[index:index + M]))

# Sort both in descending order
B.sort(reverse=True)
W.sort(reverse=True)

# Cumulative sum for B
cumB = [0] * (N + 1)
for s in range(1, N + 1):
    cumB[s] = cumB[s - 1] + B[s - 1]

# Suffix max for B
suffix_max_B = [0] * (N + 1)
suffix_max_B[N] = cumB[N]
for K in range(N - 1, -1, -1):
    suffix_max_B[K] = max(cumB[K], suffix_max_B[K + 1])

# Cumulative sum for W
cumW = [0] * (M + 1)
for k in range(1, M + 1):
    cumW[k] = cumW[k - 1] + W[k - 1]

# Min of M and N
min_M_N = min(M, N)

# Find the maximum sum
ans = float('-inf')
for K in range(0, min_M_N + 1):
    total = cumW[K] + suffix_max_B[K]
    if total > ans:
        ans = total

print(ans)