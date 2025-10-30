N = int(input())
A = list(map(int, input().split()))

# Precompute prefix distinct counts
prefix_distinct = [0] * N
seen = [False] * (N + 1)
distinct_count = 0
for i in range(N):
    if not seen[A[i]]:
        seen[A[i]] = True
        distinct_count += 1
    prefix_distinct[i] = distinct_count

# Precompute suffix distinct counts
suffix_distinct = [0] * N
seen = [False] * (N + 1)
distinct_count = 0
for i in range(N-1, -1, -1):
    if not seen[A[i]]:
        seen[A[i]] = True
        distinct_count += 1
    suffix_distinct[i] = distinct_count

max_sum = 0

# Try all possible (i, j) pairs
for i in range(N-2):  # i from 0 to N-3 (0-indexed)
    middle_seen = [False] * (N + 1)
    middle_distinct = 0
    for j in range(i+1, N-1):  # j from i+1 to N-2 (0-indexed)
        if not middle_seen[A[j]]:
            middle_seen[A[j]] = True
            middle_distinct += 1
        
        # Calculate total: first + middle + third subarray distinct counts
        total = prefix_distinct[i] + middle_distinct + suffix_distinct[j+1]
        if total > max_sum:
            max_sum = total

print(max_sum)