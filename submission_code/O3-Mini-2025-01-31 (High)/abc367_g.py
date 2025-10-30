# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    K = int(next(it))
    mod = 998244353
    A_list = [int(next(it)) for _ in range(N)]
    
    # -------------------------------
    # Compute a GF(2) linear basis from A.
    # Since A[i] < 2^20, the number of bits is 20.
    basis = [0] * 20
    for a in A_list:
        x = a
        for i in range(19, -1, -1):
            if (x >> i) & 1:
                if basis[i]:
                    x ^= basis[i]
                else:
                    basis[i] = x
                    break
    B = [x for x in basis if x]
    r = len(B)
    if r == 0:
        sys.stdout.write("0")
        return
    # To get a systematic representation, sort B by bit–length (so that B[0] becomes coordinate–bit 0, etc.)
    B.sort(key=lambda x: x.bit_length())
    # Define a function get_coord(x) that computes the coordinate (an integer in [0,2^r)) for x relative to B.
    def get_coord(x):
        coord = 0
        for i, b in enumerate(B):
            p = b.bit_length() - 1
            if (x >> p) & 1:
                x ^= b
                coord |= (1 << i)
        return coord
    
    # Build frequency dictionary: for each A[i] compute its coordinate representation.
    freq = {}
    for a in A_list:
        c = get_coord(a)
        freq[c] = freq.get(c, 0) + 1

    # -------------------------------
    # Precompute factorials up to N so we can compute binomials quickly.
    max_n = N
    fact = [1] * (max_n+1)
    invfact = [1] * (max_n+1)
    for i in range(2, max_n+1):
        fact[i] = fact[i-1] * i % mod
    invfact[max_n] = pow(fact[max_n], mod-2, mod)
    for i in range(max_n, 0, -1):
        invfact[i-1] = invfact[i] * i % mod
    def binom(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * invfact[k] % mod * invfact[n-k] % mod

    # For a count c, we compute two length–M polys:
    #   pe[j] = sum_{j' even, j' mod M == j} C(c,j')
    #   po[j] = sum_{j' odd, j' mod M == j}  C(c,j')
    def compute_poly(c):
        pe = [0]*M
        po = [0]*M
        for j in range(c+1):
            bj = binom(c, j)
            rmod = j % M
            if j & 1:
                po[rmod] = (po[rmod] + bj) % mod
            else:
                pe[rmod] = (pe[rmod] + bj) % mod
        return pe, po

    # Precompute for every distinct coordinate (group) the polynomial pair.
    group_poly = {}
    for key, cnt in freq.items():
        group_poly[key] = compute_poly(cnt)
    
    # -------------------------------
    # We now wish to “multiply” these group–factors.
    # Our “XOR–domain” is the set of r–bit vectors; its size is L_size = 2^r.
    L_size = 1 << r
    # We represent a function on the XOR domain as a list F of length L_size,
    # where F[u] is a poly (list of length M) giving the total number of ways
    # to have a subsequence with coordinate XOR = u and with size ≡ d (mod M).
    F_poly = [[0]*M for _ in range(L_size)]
    F_poly[0][0] = 1  # empty subset

    # Helper: cyclic convolution (polynomial multiplication mod x^M-1) of two polys (length M).
    def poly_mul(a, b):
        res = [0]*M
        for i in range(M):
            ai = a[i]
            if ai:
                for j in range(M):
                    res[(i+j) % M] = (res[(i+j) % M] + ai * b[j]) % mod
        return res
    def poly_add(a, b):
        return [(a[i] + b[i]) % mod for i in range(M)]
    
    # “Multiply” F by a group factor. For a group with key g and factor (pe,po)
    # the rule is: for each u in the XOR–domain, 
    # new_F[u] = F[u] * pe   +   F[u XOR g] * po
    # (where polynomial multiplication here means a cyclic convolution mod M).
    def conv_with_group(F, g, pe, po):
        newF = [[0]*M for _ in range(L_size)]
        for u in range(L_size):
            # First part: from F[u] with no XOR shift.
            part1 = poly_mul(F[u], pe)
            v = u ^ g
            part2 = poly_mul(F[v], po)
            newF[u] = [(part1[i] + part2[i]) % mod for i in range(M)]
        return newF

    # Multiply in all the group factors from group_poly.
    for g, (pe, po) in group_poly.items():
        F_poly = conv_with_group(F_poly, g, pe, po)
    
    # -------------------------------
    # Now F_poly[u][d] is the number of subsequences S ⊆ A (including empty S)
    # with “coordinate XOR” = u and with |S| ≡ d (mod M).
    # We want the sum over all nonempty S with |S| ≡ 0 mod M of (XOR(S))^K.
    # (The empty subsequence is ignored because K>=1.)
    # If u is a coordinate then its “real” XOR value is computed by
    # taking the XOR of those B[i] for which the i–th bit is set. 
    def coord_to_val(u):
        res = 0
        idx = 0
        while u:
            if u & 1:
                res ^= B[idx]
            u //= 2
            idx += 1
        return res

    ans = 0
    for u in range(L_size):
        ways = F_poly[u][0]
        if u == 0:
            ways = (ways - 1) % mod  # subtract empty set from u=0
        if ways:
            # Compute the “real” XOR value from the coordinate u.
            val = coord_to_val(u)
            # Then add (val)^K * ways, mod mod.
            ans = (ans + pow(val, K, mod) * ways) % mod
    sys.stdout.write(str(ans % mod))
    
if __name__ == '__main__':
    main()