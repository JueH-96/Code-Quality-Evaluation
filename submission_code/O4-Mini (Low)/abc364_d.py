import sys
import threading

def main():
    import sys
    from bisect import bisect_left, bisect_right

    input = sys.stdin.readline

    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    A.sort()

    out = []
    for _ in range(Q):
        b, k = map(int, input().split())
        # We want the smallest D >= 0 such that
        # number of A_i with |A_i - b| <= D is >= k.
        lo, hi = 0, max(abs(A[0] - b), abs(A[-1] - b))
        # Binary search on D
        while lo < hi:
            mid = (lo + hi) // 2
            # count A_i in [b - mid, b + mid]
            left = bisect_left(A, b - mid)
            right = bisect_right(A, b + mid)
            if right - left >= k:
                hi = mid
            else:
                lo = mid + 1
        out.append(str(lo))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()