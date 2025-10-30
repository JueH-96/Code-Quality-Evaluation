import bisect

n = int(input())
A = list(map(int, input().split()))

total = 0
for j in range(n):
    target = A[j] // 2
    count = bisect.bisect_right(A, target, 0, j)
    total += count

print(total)