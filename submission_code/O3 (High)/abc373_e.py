import sys, bisect

def main() -> None:
    input = sys.stdin.readline

    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    tot = sum(A)
    R = K - tot                      # votes that are still uncounted

    # Trivial case : everyone is always elected
    if M == N:
        print(' '.join(['0'] * N))
        return

    # sort votes in ascending order, remember original positions
    asc = sorted((v, idx) for idx, v in enumerate(A))
    A_sorted = [v for v, _ in asc]                  # ascending votes
    pos_in_sorted = [0] * N                         # original -> position in sorted
    for p, (_, idx) in enumerate(asc):
        pos_in_sorted[idx] = p

    # prefix sums of the ascending sequence
    pref = [0] * (N + 1)
    for i, v in enumerate(A_sorted, 1):
        pref[i] = pref[i - 1] + v

    def safe(i: int, x: int) -> bool:
        """Is candidate i surely elected if she receives x additional votes ?"""
        t = A[i] + x                              # her final tally
        idxT = bisect.bisect_right(A_sorted, t)   # first index with vote  > t
        hi = N - idxT                             # already ahead of i
        if hi >= M:                               # too many strictly higher
            return False

        L = M - hi                                # still need this many to pass i
        if L == 0:                                # nobody can be made to surpass
            return True

        seg_size = idxT                           # count with votes <= t
        low_others = seg_size - 1                 # those not ahead and not i

        # if not enough opponents exist to form M candidates, i is safe
        if low_others < L:
            return True

        pos = pos_in_sorted[i]

        # sum of the L largest votes among opponents with vote <= t
        start_last_L = idxT - L
        if pos < start_last_L:                    # i not in that suffix
            s_top = pref[idxT] - pref[start_last_L]
        else:                                     # i lies inside that suffix
            # take L+1 largest (includes i) and subtract i's own vote
            s_total = pref[idxT] - pref[idxT - L - 1]
            s_top = s_total - A[i]

        need = L * (t + 1) - s_top                # votes adversary must add
        return need > R - x                       # secure if adversary lacks votes

    res = [0] * N
    for i in range(N):
        if safe(i, 0):              # already unassailable
            res[i] = 0
            continue

        if R == 0 or not safe(i, R):  # even all remaining votes are not enough
            res[i] = -1
            continue

        lo, hi = 0, R               # binary-search smallest x with safe(i, x)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if safe(i, mid):
                hi = mid
            else:
                lo = mid
        res[i] = hi

    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()