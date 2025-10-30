import sys
import threading
def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = 1 << N

    # summs[mask] = sum of A[i] for bits i in mask
    summs = [0] * M
    # lsb_idx[mask] = the index (0-based) of the least significant set bit of mask
    lsb_idx = [0] * M
    for i in range(N):
        lsb_idx[1 << i] = i
    for mask in range(1, M):
        f = mask & -mask
        idx = lsb_idx[f]
        lsb_idx[mask] = idx
        summs[mask] = summs[mask ^ f] + A[idx]

    # For each mask, we will enumerate all submasks t of (mask without its LSB).
    subs = [[] for _ in range(M)]
    for mask in range(1, M):
        f = mask & -mask
        R = mask ^ f
        lst = subs[mask]
        t = R
        # enumerate all submasks of R
        while True:
            lst.append(t)
            if t == 0:
                break
            t = (t - 1) & R

    # dp[mask] = set of all possible XOR-results obtainable by partitioning the bits in `mask`
    dp = [None] * M
    dp[0] = {0}

    sm = summs
    subsls = subs
    dp_local = dp

    for mask in range(1, M):
        tot = sm[mask]
        new_set = set()
        new_add = new_set.add
        # We fix the block containing the LSB of mask; the rest of that block is
        # any submask `t` of (mask without LSB).  Then we combine dp[t] with
        # the XOR of the one chosen block = (tot - sm[t]).
        for t in subsls[mask]:
            delta = tot - sm[t]
            for x in dp_local[t]:
                new_add(x ^ delta)
        dp_local[mask] = new_set

    ans = len(dp_local[M - 1])
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()