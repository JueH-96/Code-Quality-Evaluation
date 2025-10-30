import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

def count_sellers(x):
    return len(A) - bisect.bisect_left(A, x)

def count_buyers(x):
    return bisect.bisect_right(B, x)

low = 1
high = 10**18  # A value can be up to 1e9, so 1e18 is safe

while low < high:
    mid = (low + high) // 2
    s = count_sellers(mid)
    b = count_buyers(mid)
    if s >= b:
        high = mid
    else:
        low = mid + 1

print(low)