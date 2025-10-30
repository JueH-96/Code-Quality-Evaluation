import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # Precompute prefix sums for O(1) range-sum queries
    S = [0] * (N + 1)
    for i in range(N):
        S[i+1] = S[i] + A[i]

    def range_sum(l, r):
        # inclusive l..r
        return S[r+1] - S[l]

    # For each center we will expand its interval [l, r] by the two‑pointer
    # consuming process described.  We can amortize to O(N) total by
    # remembering that once two intervals meet they won't separate.
    #
    # We'll maintain for each position i its final interval [L[i], R[i]]
    # and final sum B[i].  We fill these by scanning i from left to right,
    # and whenever we finish i, we "paint" the entire interval so
    # subsequent centers inside need not re-simulate.
    L = [-1] * N
    R = [-1] * N
    B = [0] * N

    for i in range(N):
        if L[i] != -1:
            # already computed via a previous center covering us
            continue
        # begin a fresh expansion from center i
        l = i
        r = i
        cur = A[i]
        # expand until stuck
        while True:
            moved = False
            # try push left
            if l > 0 and cur > A[l-1]:
                l -= 1
                cur += A[l]
                moved = True
                continue
            # try push right
            if r < N-1 and cur > A[r+1]:
                r += 1
                cur += A[r]
                moved = True
                continue
            if not moved:
                break
        # now [l,r] is final for this center, cur is total sum
        # paint all inside
        for j in range(l, r+1):
            L[j] = l
            R[j] = r
            B[j] = cur

    # output
    print(" ".join(str(B[i]) for i in range(N)))

if __name__ == "__main__":
    main()