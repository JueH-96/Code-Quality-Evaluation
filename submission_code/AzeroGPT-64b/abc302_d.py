from bisect import bisect_left

N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
A.sort()
B.sort()

maxVal = -1
for a in A:
    lower = bisect_left(B, a - D)
    upper = bisect_left(B, a + D + 1)
    if lower != M and lower != upper:
        maxVal = max(maxVal, a + B[lower])
    if lower != M and lower + 1 != upper:
        maxVal = max(maxVal, a + B[lower + 1])

print(maxVal)