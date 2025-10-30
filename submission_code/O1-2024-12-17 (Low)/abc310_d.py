def main():
    import sys
    sys.setrecursionlimit(10**7)

    # Read inputs
    data = sys.stdin.read().strip().split()
    N, T, M = map(int, data[:3])
    pairs = data[3:]
    incompat = [[False]*N for _ in range(N)]
    idx = 0
    for _ in range(M):
        a = int(pairs[idx]) - 1  # convert to 0-based
        b = int(pairs[idx+1]) - 1
        idx += 2
        incompat[a][b] = True
        incompat[b][a] = True

    # Precompute valid[mask]: True if no incompatible pair is within this subset
    # A subset is represented by a bitmask from 0 to (1<<N)-1
    valid = [True]*(1<<N)
    for mask in range(1, 1<<N):
        # Build list of players in this mask
        # Then check any pair among them
        players = []
        temp = mask
        bit_idx = 0
        while temp > 0:
            if temp & 1:
                players.append(bit_idx)
            bit_idx += 1
            temp >>= 1
        # Check incompatible pairs
        nsub = len(players)
        for i in range(nsub):
            for j in range(i+1, nsub):
                if incompat[players[i]][players[j]]:
                    valid[mask] = False
                    break
            if not valid[mask]:
                break

    # dp[mask][k] = number of ways to partition the set "mask" (unlabeled) into k valid subsets
    # with the restriction that each subset is "valid" (no incompatible pairs).
    dp = [[0]*(T+1) for _ in range(1<<N)]
    dp[0][0] = 1  # Base case

    # Helper to get lowest set bit index
    def lowest_set_bit(x):
        return (x & -x).bit_length() - 1  # or x.bit_length()-1 for x being a power of two

    for mask in range(1, 1<<N):
        # We'll fill dp[mask][k] for k=1..T
        lb = lowest_set_bit(mask)  # index of lowest set bit in mask
        # remove lb from mask for sub-subset enumeration
        mask_without_lb = mask ^ (1 << lb)
        # Enumerate all sub-subsets that do contain lb (through that bit).
        sub = mask_without_lb
        while True:
            # sub is a subset of mask_without_lb
            subset_with_lb = sub | (1 << lb)
            # We consider subset_with_lb as one "block" if valid
            if valid[subset_with_lb]:
                # We add ways from dp[mask ^ subset_with_lb][k-1]
                leftover = mask ^ subset_with_lb
                for k in range(1, T+1):
                    dp[mask][k] += dp[leftover][k-1]

            if sub == 0:
                break
            sub = (sub - 1) & mask_without_lb

    print(dp[(1<<N) - 1][T])

# Call main
if __name__ == "__main__":
    main()