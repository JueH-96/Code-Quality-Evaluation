# YOUR CODE HERE
def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    # parse input
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # Compute a linear basis (over GF(2)) for the sequence A.
    # (A standard “Gaussian elimination” in GF(2).)
    basis = []
    for a in A:
        x = a
        for b in basis:
            nx = x ^ b
            if nx < x:
                x = nx
        if x:
            basis.append(x)
    # sort the basis (the order does not matter but fix an order)
    basis.sort(reverse=True)
    r = len(basis)
    f = N - r  # number of free elements (redundant ones)
    
    # Precompute factorials and inverse factorials up to f (to compute binomials C(f,j) mod mod)
    fact = [1]*(f+1)
    invfact = [1]*(f+1)
    for i in range(1, f+1):
        fact[i] = fact[i-1] * i % mod
    invfact[f] = pow(fact[f], mod-2, mod)
    for i in range(f, 0, -1):
        invfact[i-1] = invfact[i] * i % mod
        
    # Precompute for each j = 0 … f the binomial C(f,j)
    # And then accumulate, for each residue mod M,
    # R[r0] = sum{ C(f, j) for j such that j mod M == r0 }
    R = [0]*M
    for j in range(f+1):
        # binom = C(f, j)
        binom = fact[f] * invfact[j] % mod * invfact[f - j] % mod
        R[j % M] = (R[j % M] + binom) % mod
    # For any subset S (of the basis) with |S| = k, the free–element choices sum to
    # F(k) = ∑_{j: (k+j) mod M == 0} C(f,j)  = R[(-k) mod M].
    T = [0]*(r+1)
    for k in range(r+1):
        T[k] = R[(-k) % M]
        
    # Now every nonempty subset S of the basis produces an overall subsequence
    # with XOR = (xor of elements in S) and total size = |S| + j (where the free part j may be chosen arbitrarily).
    # And the score for that subsequence is (xor)^K.
    # (If S is empty, XOR = 0 and 0^K=0; so we ignore S==∅.)
    # We iterate over all subsets of the basis.
    size = 1 << r
    dp_xor = [0] * size   # dp_xor[mask] will hold the XOR from the chosen basis elements.
    dp_pop = [0] * size   # dp_pop[mask] = number of basis elements chosen.
    ans = 0
    # We enumerate all nonempty masks from 1 to 2^r - 1.
    # We use the recurrence: let lb = mask & -mask, and let its index be i,
    # then dp_xor[mask] = dp_xor[mask ^ lb] XOR basis[i] and similarly for the popcount.
    for mask in range(1, size):
        lb = mask & -mask
        i = lb.bit_length() - 1
        prev = mask ^ lb
        dp_xor[mask] = dp_xor[prev] ^ basis[i]
        dp_pop[mask] = dp_pop[prev] + 1
        # Add the contribution for this choice:
        # (xor)^K * [free-part sum corresponding to dp_pop[mask]].
        ans = (ans + pow(dp_xor[mask], K, mod) * T[dp_pop[mask]]) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()