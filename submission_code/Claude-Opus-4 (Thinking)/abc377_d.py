# YOUR CODE HERE
N, M = map(int, input().split())
intervals = []
for _ in range(N):
    L, R = map(int, input().split())
    intervals.append((L, R))

# For each l, precompute min(R_i for all intervals with L_i >= l)
min_r_from_l = [M + 1] * (M + 2)

for L_i, R_i in intervals:
    min_r_from_l[L_i] = min(min_r_from_l[L_i], R_i)

# Propagate the minimum values
for l in range(M, 0, -1):
    if l < M:
        min_r_from_l[l] = min(min_r_from_l[l], min_r_from_l[l + 1])

count = 0
for l in range(1, M + 1):
    max_valid_r = min(M, min_r_from_l[l] - 1)
    if max_valid_r >= l:
        count += max_valid_r - l + 1

print(count)