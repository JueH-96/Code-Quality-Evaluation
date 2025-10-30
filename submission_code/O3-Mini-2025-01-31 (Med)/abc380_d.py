def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    # The first token is the initial string S.
    S = data[0].decode() if isinstance(data[0], bytes) else data[0]
    n = len(S)
    
    # Next token is the number of queries Q.
    Q = int(data[1])
    
    # Next tokens are the query values K.
    # (They can be huge, up to 10^18)
    queries = [int(x) for x in data[2:2+Q]]
    
    # A small helper: swap-case for one character.
    def swap_char(c):
        return c.swapcase()
    
    # We will build our answer for every query.
    # For each query, we want to simulate the reverse of the operations.
    # Instead of using recursion, we iterate:
    #   Let L = n * 2^d, where L is the smallest such length >= k.
    #   Then while L > n:
    #       if k is in second half (i.e. > L/2), adjust:
    #            k = k - L/2 and toggle parity.
    #       Reduce L by half.
    #   Then answer = S[k-1] (swapped if parity indicates).
    #
    # Note: Each query takes O(log(K)) ~ O(60) steps.
    
    out = []
    
    for orig_k in queries:
        k = orig_k
        # r indicates swap parity (0 = no swap, 1 = swap).
        r = 0
        
        # If k is within the initial string, we immediately answer.
        if k <= n:
            ch = S[k-1]
            if r: 
                ch = swap_char(ch)
            out.append(ch)
            continue
        
        # Compute d such that L = n << d (i.e. n * 2^d) is >= k.
        d = 0
        if k > n:
            # Using bit_length, we can get an initial guess.
            d = (k - 1).bit_length() - n.bit_length() + 1
            while (n << d) < k:
                d += 1
        L = n << d  # L = n * 2^d
        
        # Iteratively determine which half k is in.
        while d:
            half = L >> 1  # L/2
            if k > half:
                # k is in the second half: subtract and toggle the flip.
                k -= half
                r ^= 1
            L = half
            d -= 1
        
        # Now L == n, so k is within the original S.
        ch = S[k-1]
        if r:
            ch = swap_char(ch)
        out.append(ch)
    
    sys.stdout.write(" ".join(out))

if __name__ == '__main__':
    main()