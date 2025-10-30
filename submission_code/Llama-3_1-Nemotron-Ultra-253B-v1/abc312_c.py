import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

max_X = max(A[-1] if n else 0, B[-1] if m else 0)
low = 1
high = max_X

while low < high:
    mid = (low + high) // 2
    sellers = bisect.bisect_right(A, mid)
    buyers = m - bisect.bisect_left(B, mid)
    if sellers >= buyers:
        high = mid
    else:
        low = mid + 1

print(low)