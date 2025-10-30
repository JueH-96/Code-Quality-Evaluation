N, M = map(int, input().split())
B = list(map(int, input().split()))
W = list(map(int, input().split()))

# Sort in descending order to prioritize higher values
B.sort(reverse=True)
W.sort(reverse=True)

# Compute prefix sums
black_prefix = [0] * (N + 1)
for i in range(N):
    black_prefix[i + 1] = black_prefix[i] + B[i]

white_prefix = [0] * (M + 1)
for i in range(M):
    white_prefix[i + 1] = white_prefix[i] + W[i]

# Compute suffix maximum of black prefix sums
# black_suffix_max[i] = maximum sum achievable using at least i black balls
black_suffix_max = [0] * (N + 1)
for i in range(N, -1, -1):
    if i == N:
        black_suffix_max[i] = black_prefix[i]
    else:
        black_suffix_max[i] = max(black_suffix_max[i + 1], black_prefix[i])

# Try all possible numbers of white balls
max_sum = 0
for k in range(min(M + 1, N + 1)):  # k white balls requires at least k black balls
    white_sum = white_prefix[k]
    black_sum = black_suffix_max[k]  # Best sum using at least k black balls
    total = white_sum + black_sum
    max_sum = max(max_sum, total)

print(max_sum)