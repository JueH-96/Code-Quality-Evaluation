def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    mod = 998244353

    # We wish to compute:
    #    S = sum_{k=0}^{N} popcount(k & M)
    #
    # For each k, popcount(k & M) is the number of bit positions j for which both k and M have a 1 in that position.
    # This allows us to change the order of summation:
    #    S = sum_{j such that M has a 1 in position j} (the number of k in [0,N] whose j-th bit is set)
    #
    # For a fixed j, the j-th bit in k follows a periodic pattern with period = 2^(j+1)
    # and the bit is 1 for 2^j consecutive numbers in that period.
    # Thus, if total = N+1 numbers from 0 to N,
    #   count for bit j = (total // (2^(j+1)))*2^j + max(0, (total % (2^(j+1)) - 2^j)).
    
    total = N + 1
    ans = 0
    # Check bits from 0 up to 60. (Since N, M are at most 2^60 - 1.)
    for j in range(61):
        if (M >> j) & 1:
            cycle = 1 << (j + 1)
            half = 1 << j
            full_cycles = total // cycle
            remainder = total % cycle
            count = full_cycles * half + max(0, remainder - half)
            ans = (ans + count) % mod
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()