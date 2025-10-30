import bisect

N = int(input())
A = list(map(int, input().split()))

# Create a sorted version of A
sorted_A = sorted(A)

# Calculate suffix sums (sum from index i to end)
suffix_sum = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sum[i] = suffix_sum[i + 1] + sorted_A[i]

results = []
for i in range(N):
    # Find the first position where elements are greater than A[i]
    pos = bisect.bisect_right(sorted_A, A[i])
    # Sum of elements greater than A[i]
    sum_greater = suffix_sum[pos]
    results.append(sum_greater)

print(' '.join(map(str, results)))