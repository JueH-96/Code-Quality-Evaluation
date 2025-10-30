def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    mod = 998244353

    # Determine the bit length we need to consider.
    L = max(N.bit_length(), M.bit_length())
    # Extract bits (from most-significant to least-significant)
    N_bits = [(N >> i) & 1 for i in range(L-1, -1, -1)]
    M_bits = [(M >> i) & 1 for i in range(L-1, -1, -1)]

    from functools import lru_cache

    # dp(pos, tight) returns a tuple:
    # (number of ways to form a valid k from pos to L-1, 
    #  total sum of contributions (popcount) from bits chosen on positions pos...L-1)
    # "tight" indicates whether the prefix so far is equal to N's prefix (tight==True) or already lower.
    @lru_cache(maxsize=None)
    def dp(pos, tight):
        if pos == L:
            return (1, 0)  # one number formed and its contribution sum is 0
        ways = 0
        ssum = 0
        # Determine the upper limit for k's current bit.
        limit = N_bits[pos] if tight else 1
        for bit in [0, 1]:
            if bit > limit:
                continue  # skip invalid bit choice that would make k exceed N.
            new_tight = tight and (bit == limit)
            count, sub_sum = dp(pos + 1, new_tight)
            # Extra contribution at this bit: if bit is 1 and corresponding M bit is also 1.
            extra = bit & M_bits[pos]
            ways = (ways + count) % mod
            ssum = (ssum + sub_sum + extra * count) % mod
        return (ways, ssum)

    result = dp(0, True)[1] % mod
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()