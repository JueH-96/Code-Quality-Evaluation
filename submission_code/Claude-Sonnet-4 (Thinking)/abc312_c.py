import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

def sellers_count(x):
    return bisect.bisect_right(A, x)

def buyers_count(x):
    return m - bisect.bisect_left(B, x)

# Critical points where the counts change
critical = set(A + [b + 1 for b in B])
critical = sorted(critical)

for x in critical:
    if sellers_count(x) >= buyers_count(x):
        print(x)
        break