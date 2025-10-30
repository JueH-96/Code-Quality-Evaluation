import sys
import threading

def main():
    import sys
    import bisect

    input = sys.stdin.readline
    N, M, D = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    A.sort()
    B.sort()

    ans = -1
    for a in A:
        lo = a - D
        hi = a + D
        # Find rightmost index in B with B[idx] <= hi
        idx = bisect.bisect_right(B, hi) - 1
        if idx >= 0 and B[idx] >= lo:
            s = a + B[idx]
            if s > ans:
                ans = s

    print(ans)

if __name__ == "__main__":
    main()