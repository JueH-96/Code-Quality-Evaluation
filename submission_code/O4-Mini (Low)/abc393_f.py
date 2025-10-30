import sys
import threading
def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    A = list(map(int, input().split()))

    # Read queries, store (R, X, idx)
    queries_at = [[] for _ in range(N+1)]
    Xs = []
    for qi in range(Q):
        r, x = map(int, input().split())
        queries_at[r].append((x, qi))
        Xs.append(x)

    # Coordinate compress the A values
    vals = sorted(set(A))
    # helper to map an A[i] to 1-based index
    import bisect
    def coord(v):
        # returns 1..M
        return bisect.bisect_left(vals, v) + 1

    M = len(vals)

    # BIT for range [1..M], supporting point update max and prefix max query
    class BITMax:
        def __init__(self, n):
            self.n = n
            self.data = [0] * (n+1)
        def update(self, i, v):
            # set data[i] = max(data[i], v)
            while i <= self.n:
                if self.data[i] < v:
                    self.data[i] = v
                else:
                    # if current already >= v, higher nodes already >= as well
                    # but we must still go up because maybe they were lower
                    # we skip early only if data[i] >= v and no need propagate
                    # however safe to continue
                    pass
                i += i & -i
        def query(self, i):
            # max on [1..i]
            res = 0
            while i > 0:
                if self.data[i] > res:
                    res = self.data[i]
                i -= i & -i
            return res

    bit = BITMax(M)

    ans = [0]*Q

    # Process positions 1..N
    for i in range(1, N+1):
        ai = A[i-1]
        ci = coord(ai)
        # dp[i] = 1 + max dp[j] with A[j] < A[i]
        best_before = bit.query(ci-1)
        dpi = best_before + 1
        # update at value A[i]
        bit.update(ci, dpi)

        # answer queries with R == i
        if queries_at[i]:
            for x, qi in queries_at[i]:
                # find highest compressed index with value <= x
                # that is bisect_right(vals, x)
                cx = bisect.bisect_right(vals, x)
                if cx == 0:
                    ans[qi] = 0
                else:
                    ans[qi] = bit.query(cx)

    # output
    out = sys.stdout
    for v in ans:
        out.write(str(v) + "
")

if __name__ == "__main__":
    main()