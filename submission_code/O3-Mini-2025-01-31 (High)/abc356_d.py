def main():
    import sys
    mod = 998244353

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    # The idea:
    # popcount(k & M) = sum_{i: (M >> i & 1) == 1} [the i-th bit of k].
    # So the total sum = sum_{k=0}^{N} popcount(k & M)
    #                = sum_{i such that M has a 1 in bit i} (number of k in [0,N] with the i-th bit = 1).
    #
    # For a given bit i, the number of numbers between 0 and N (inclusive) having bit i set can be computed
    # by a well-known formula. Let L = N + 1. Then:
    # 
    #   count_i = (L // (2^(i+1))) * (2^i) + max(0, L % (2^(i+1)) - 2^i).
    #
    # We then add count_i for every i such that the i-th bit of M is 1, and take the answer mod 998244353.
    
    total = 0
    # N and M are at most 2^60-1, so we only need to check bits i=0,...,60.
    NP1 = N + 1
    for i in range(61):
        if (M >> i) & 1:
            # For bit i, compute the number of k in [0, N] such that the i-th bit is set.
            full_cycles = NP1 // (1 << (i + 1))
            ones_from_cycles = full_cycles * (1 << i)
            remainder = NP1 % (1 << (i + 1))
            ones_from_partial = remainder - (1 << i) if remainder > (1 << i) else 0
            count_i = ones_from_cycles + ones_from_partial
            total = (total + count_i) % mod
            
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()