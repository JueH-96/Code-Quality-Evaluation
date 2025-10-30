import sys
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    T = int(next(it))
    M = int(next(it))
    badmask = [0]*N
    for _ in range(M):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        badmask[a] |= 1<<b
        badmask[b] |= 1<<a

    sz = 1 << N
    # valid[mask] = True iff mask has no incompatible pair
    valid = [False] * sz
    valid[0] = True
    for mask in range(1, sz):
        ok = True
        m = mask
        # check each bit i in mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            # if any incompatible neighbor of i in mask => invalid
            if (mask & badmask[i]) != 0:
                ok = False
                break
            m ^= lsb
        valid[mask] = ok

    # submasks_list[mask] = list of nonempty submasks of mask
    # that contain the least-significant bit of mask and are valid
    submasks_list = [[] for _ in range(sz)]
    for mask in range(1, sz):
        lm = mask & -mask
        sub = mask
        while True:
            # include sub if it contains lm and is valid
            if (sub & lm) and valid[sub]:
                submasks_list[mask].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & mask

    # dp[mask][k] = number of ways to partition mask into k blocks
    dp = [ [0]*(T+1) for _ in range(sz) ]
    dp[0][0] = 1

    for mask in range(1, sz):
        # compute dp[mask][k] for k=1..T
        for k in range(1, T+1):
            total = 0
            for sub in submasks_list[mask]:
                total += dp[mask ^ sub][k-1]
            dp[mask][k] = total

    ans = dp[sz-1][T]
    print(ans)

if __name__ == "__main__":
    main()