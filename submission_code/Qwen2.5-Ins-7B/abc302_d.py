# YOUR CODE HERE
import bisect

def solve():
    import sys
    input = sys.stdin.readline

    N, M, D = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    A.sort()
    B.sort()

    max_sum = -1
    for a in A:
        if a > 10**18 - D:
            break
        idx = bisect.bisect_right(B, a + D)
        if idx > 0:
            max_sum = max(max_sum, a + B[idx-1])

    print(max_sum)

solve()