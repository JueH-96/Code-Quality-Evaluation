import bisect

n = int(input())
A = list(map(int, input().split()))
sorted_A = sorted(A)
suffix_sum = [0] * n

# Compute suffix sum
suffix_sum[-1] = sorted_A[-1]
for i in range(n - 2, -1, -1):
    suffix_sum[i] = sorted_A[i] + suffix_sum[i + 1]

result = []
for a in A:
    idx = bisect.bisect_right(sorted_A, a)
    if idx < n:
        result.append(str(suffix_sum[idx]))
    else:
        result.append('0')

print(' '.join(result))