import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    Q = int(next(it))
    out = []
    for _ in range(Q):
        L = int(next(it)) - 1
        R = int(next(it)) - 1
        length = R - L + 1
        lo, hi = 0, length // 2
        best = 0
        # Binary search on K = number of pairs
        while lo <= hi:
            mid = (lo + hi) // 2
            ok = True
            # Check if we can form 'mid' pairs:
            # For i in [0..mid-1], A[L + i] * 2 <= A[R - mid + 1 + i]
            base2 = R - mid + 1
            for i in range(mid):
                if A[L + i] * 2 > A[base2 + i]:
                    ok = False
                    break
            if ok:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out.append(str(best))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()