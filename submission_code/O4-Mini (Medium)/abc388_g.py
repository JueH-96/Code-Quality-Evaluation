import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    q = int(next(it))
    out = []
    for _ in range(q):
        L = int(next(it)) - 1
        R = int(next(it)) - 1
        M = R - L + 1
        # binary search maximum K in [0..M//2]
        lo, hi = 0, M // 2
        # check function
        while lo < hi:
            mid = (lo + hi + 1) // 2
            # can we make mid pairs?
            ok = True
            # pair i-th smallest of segment with i-th largest
            # small indices L .. L+mid-1, large indices R-mid+1 .. R
            # check A[L+i]*2 <= A[R-mid+1+i]
            a = L
            b = R - mid + 1
            for i in range(mid):
                if A[a + i] * 2 > A[b + i]:
                    ok = False
                    break
            if ok:
                lo = mid
            else:
                hi = mid - 1
        out.append(str(lo))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()