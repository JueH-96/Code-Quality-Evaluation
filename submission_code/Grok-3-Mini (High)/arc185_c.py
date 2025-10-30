import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1
A = list(map(int, data[index:index+N]))

# Precompute suffix min and max
if N > 0:
    suffix_min = [0] * N
    suffix_max = [0] * N
    suffix_min[N-1] = A[N-1]
    suffix_max[N-1] = A[N-1]
    for p in range(N-2, -1, -1):
        suffix_min[p] = min(A[p], suffix_min[p+1])
        suffix_max[p] = max(A[p], suffix_max[p+1])

for i in range(0, N-2):
    target = X - A[i]
    min_val = suffix_min[i+1]
    max_val = suffix_max[i+1]
    min_sum_pair = 2 * min_val
    max_sum_pair = 2 * max_val
    if min_sum_pair > target or max_sum_pair < target:
        continue  # no pair possible
    # do two-sum on A[i+1:]
    seen = {}
    for idx in range(i+1, N):
        num = A[idx]
        complement = target - num
        if complement in seen:
            m = seen[complement]
            k = idx
            # output 1-based indices
            print(f"{i+1} {m+1} {k+1}")
            sys.exit()
        if num not in seen:
            seen[num] = idx  # add first occurrence of the value

# no triple found
print("-1")