import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N-1)]

    A.sort()
    B.sort()

    # f(k): if we remove A[k], can we match the remaining N-1 toys to the N-1 boxes B?
    def f(k):
        # two pointers over A and B, skipping index k in A
        i = 0
        n = N
        for b in B:
            # skip the removed toy
            if i == k:
                i += 1
            # if we ran out of toys, fail
            if i >= n:
                return False
            # the smallest remaining toy must fit in this box
            if b < A[i]:
                return False
            # match and move to next toy
            i += 1
        # matched all boxes
        return True

    # binary search for smallest k in [0..N-1] with f(k) == True
    lo, hi = 0, N  # hi is exclusive
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid + 1

    if lo < N and f(lo):
        print(A[lo])
    else:
        print(-1)

if __name__ == "__main__":
    main()