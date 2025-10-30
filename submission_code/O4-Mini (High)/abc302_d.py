import sys
from bisect import bisect_right

def main():
    input = sys.stdin.readline
    N, M, D = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    A.sort()
    B.sort()

    ans = -1
    # iterate over the smaller list for fewer binary searches
    if N <= M:
        for a in A:
            lo = a - D
            hi = a + D
            # find the largest b <= hi
            idx = bisect_right(B, hi) - 1
            if idx >= 0 and B[idx] >= lo:
                s = a + B[idx]
                if s > ans:
                    ans = s
    else:
        for b in B:
            lo = b - D
            hi = b + D
            idx = bisect_right(A, hi) - 1
            if idx >= 0 and A[idx] >= lo:
                s = b + A[idx]
                if s > ans:
                    ans = s

    print(ans)

main()