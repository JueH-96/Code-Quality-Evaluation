# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    S = data[0].strip()
    n = len(S)
    mod = 998244353
    # Modular inverse of 26 (26 is invertible mod 998244353)
    inv26 = pow(26, mod-2, mod)
    
    # We maintain three “groups” of states:
    # dp0[d] for state0 ("clean") for the current segment; d=0,...,26.
    # dp1 is the weight for state1 ("risky"), where a duplicate uppercase 
    #   has been seen in the current segment.
    # dp2 is the weight for state2 ("danger") – already a lowercase was seen AFTER a duplicate.
    dp0 = [0]*27
    dp0[0] = 1
    dp1 = 0
    dp2 = 0
    
    for ch in S:
        ndp0 = [0]*27
        ndp1 = 0
        ndp2 = 0
        # Process state0:
        for d in range(27):
            ways = dp0[d]
            if ways == 0:
                continue
            if ch == '?':
                # Option 1: we choose an uppercase letter.
                # There are 26 choices.
                # If the letter is new then new d becomes d+1 (if d<26)
                if d < 26:
                    ndp0[d+1] = (ndp0[d+1] + ways * (26 - d)) % mod
                # If the letter is already seen then d choices cause a duplicate → state1.
                ndp1 = (ndp1 + ways * d) % mod
                # Option 2: we choose a lowercase letter.
                # There are 26 choices and reading lowercase resets the segment (d->0)
                ndp0[0] = (ndp0[0] + ways * 26) % mod
            elif ch.isupper():
                # Fixed uppercase letter.
                # In state0 with parameter d:
                # The chance that this fixed letter is new is (26-d)/26;
                # that it is already present is d/26.
                if d < 26:
                    ndp0[d+1] = (ndp0[d+1] + ways * (26 - d) * inv26) % mod
                ndp1 = (ndp1 + ways * d * inv26) % mod
            else:
                # Fixed lowercase letter.
                # In state0 a lowercase resets the segment.
                ndp0[0] = (ndp0[0] + ways) % mod
        # Process state1:
        if ch == '?':
            # In state1: an uppercase letter (26 ways) keeps state1.
            ndp1 = (ndp1 + dp1 * 26) % mod
            # A lowercase letter (26 ways) moves state1 -> state2.
            ndp2 = (ndp2 + dp1 * 26) % mod
        elif ch.isupper():
            # Fixed uppercase: simply remain in state1.
            ndp1 = (ndp1 + dp1) % mod
        else:
            # Fixed lowercase: state1 -> state2.
            ndp2 = (ndp2 + dp1) % mod
            
        # Process state2:
        if ch == '?':
            # In state2, choosing an uppercase letter would create a forbidden pattern, so only lowercase choices (26 ways) are allowed.
            ndp2 = (ndp2 + dp2 * 26) % mod
        elif ch.isupper():
            # Fixed uppercase: branch dies (forbidden) → add 0.
            pass
        else:
            # Fixed lowercase: state2 remains state2.
            ndp2 = (ndp2 + dp2) % mod
            
        dp0, dp1, dp2 = ndp0, ndp1, ndp2

    # Our DP counts the number of completions that never entered the forbidden branch.
    # Hence the answer is the sum over all state0 + state1 + state2.
    ans = (sum(dp0) + dp1 + dp2) % mod
    sys.stdout.write(str(ans))
    
if __name__=="__main__":
    main()