# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize difference array
diff = [0] * (M + 1)

# For each pair (i, j) with i < j
for i in range(N):
    for j in range(i + 1, N):
        if A[i] > A[j]:
            # (i, j) is an inversion for k ∈ [0, M - A[i]) ∪ [M - A[j], M)
            diff[0] += 1
            diff[M - A[i]] -= 1
            diff[M - A[j]] += 1
        elif A[i] < A[j]:
            # (i, j) is an inversion for k ∈ [M - A[j], M - A[i])
            diff[M - A[j]] += 1
            diff[M - A[i]] -= 1

# Compute prefix sums to get the actual counts
inv_count = 0
for k in range(M):
    inv_count += diff[k]
    print(inv_count)