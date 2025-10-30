import bisect

n = int(input())
A = list(map(int, input().split()))
res = 0

for j in range(1, n):
    max_a = A[j] // 2
    # Find the largest i where A[i] <= max_a
    count = bisect.bisect_right(A, max_a, 0, j)
    res += count

print(res)