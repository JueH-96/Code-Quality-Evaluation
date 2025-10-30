import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N, M, D = map(int, [data[0], data[1], data[2]])
    A = list(map(int, data[3:N+3]))
    B = list(map(int, data[N+3:]))
    
    A.sort()
    B.sort()
    
    max_sum = -1
    for a in A:
        target = a + D
        idx = bisect_right(B, target)
        if idx > 0:
            b = B[idx-1]
            if abs(a - b) <= D:
                max_sum = max(max_sum, a + b)
    
    print(max_sum)

def bisect_right(a, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

solve()