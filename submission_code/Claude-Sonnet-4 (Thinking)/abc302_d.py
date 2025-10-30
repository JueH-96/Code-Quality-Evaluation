import bisect

n, m, d = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a.sort()
b.sort()

max_sum = -1

for ai in a:
    left = bisect.bisect_left(b, ai - d)
    right = bisect.bisect_right(b, ai + d)
    
    if left < right:
        bj = b[right - 1]
        max_sum = max(max_sum, ai + bj)

print(max_sum)