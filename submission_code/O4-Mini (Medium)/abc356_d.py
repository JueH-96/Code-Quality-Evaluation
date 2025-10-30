def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    MOD = 998244353

    # We want S = sum_{k=0..N} popcount(k & M).
    # Observe popcount(k & M) = sum_{i: M_i = 1} bit_i(k).
    # So S = sum_{i: M_i = 1} (# of k in [0..N] with bit i = 1).
    # For each bit i, the bits in position i cycle every 2^(i+1):
    #   in each full cycle there are 2^i ones,
    #   and in the final partial cycle there are max(0, rem - 2^i) ones,
    # where rem = (N+1) % (2^(i+1)), T = N+1.
    #
    # We'll compute each count modulo MOD and sum.

    T = N + 1
    ans = 0
    for i in range(61):  # up to bit 60
        if (M >> i) & 1:
            bit = 1 << i
            period = bit << 1
            full = T // period
            rem = T - full * period
            add = rem - bit
            if add < 0:
                add = 0
            # count_i = full*bit + add
            # take mod carefully
            bit_mod = bit % MOD
            c = (full % MOD) * bit_mod % MOD
            c = (c + (add % MOD)) % MOD
            ans = (ans + c) % MOD

    print(ans)

if __name__ == "__main__":
    main()