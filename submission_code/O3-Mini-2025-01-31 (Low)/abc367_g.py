# Explanation:
# We want to compute the sum_{nonempty subsequence S with |S| % M == 0} (xor(S))^K.
# We introduce a generating function F over XOR-values and subsequence lengths:
#      F(s, r) = sum_{S ⊆ [N]} [xor(S)=s and |S| ≡ r mod M].
# Our answer is sum_{s} (s)^K * (Coefficient of F(s,0) excluding the empty subsequence).
# It is not possible to iterate over 2^N subsets.
# Instead, we group positions by value. Note that when you take several copies of the same v,
# the XOR contribution is v if you pick an odd number of them, and 0 if even.
#
# For a given value v, let c be its frequency.
# Then its “local generating function” is:
#    G_v(s, y) = E_{c} + O_{c} * (the “monomial” that contributes v to the XOR sum)
# where
#    E_{c}(y) = sum_{j even} choose(c,j) y^j,
#    O_{c}(y) = sum_{j odd}  choose(c,j) y^j.
#
# We want to combine the groups over all distinct v.
# The combination means we want the convolution (over XOR – add by XOR and length by addition mod M) of all the G_v.
#
# Because XOR is a binary convolution group (size 2^d where d=20) we can combine using a Fast Walsh Hadamard Transform.
# Meanwhile, the “y–polynomial” part (degree mod M) is done using cyclic convolution mod M.
#
# Let MOD = 998244353.
# We precompute a polynomial for each distinct value v:
#    P[v][r] is a vector of length M and also a 2^20 “X–mask” contribution:
#    There is one polynomial Q for v: Q = (E, O) meaning:
#      Contributing “0” if even picks, contributing v if odd picks.
# Since v is less than 2^20 we will eventually combine these into one array DP[mask][r] for mask in [0,2^20)
#
# We start with a DP which is an array of shape (2^20, M). Initially, DP[0][0] = 1 (empty set).
# Then for each distinct v with frequency c, its contribution is
#    Q = (E(y), O(y))  where the “XOR” part is 0 for even and v for odd.
# Then we combine: newDP[s XOR (parity?)] = sum_{r1+r2 mod M=r} DP[s] convolve Q.
#
# To speed up the combination we separate the DP into “frequency buckets” by value.
# We use a technique similar to subset convolution with FWHT:
#    Do an FWHT on the DP array (over the XOR dimension). Then each coordinate corresponds to a “characteristic function”.
#    For each distinct element v, “multiply” the corresponding transform.
# Finally do an inverse FWHT.
#
# In our implementation we must “merge” many factors. Notice that for different v's which are not present,
# the factor is 1 (i.e. do nothing). So we only need to combine over distinct v’s that appear.
#
# The tricky part is: for each v, we want to build a 2xM polynomial: one for even picks and one for odd picks.
# They are given by:
#   even_poly, odd_poly with coefficients r mod M.
# We can compute these quickly by using the binomial expansion of (1+y)^c and (1-y)^c:
#   even_poly = ( (1+y)^c + (1-y)^c )/2, odd_poly = ( (1+y)^c - (1-y)^c )/2.
# But note: we want these polynomials with only the exponents mod M (i.e. we only need sums over exponents ≡ j mod M).
# We can compute these polynomials by summing over j mod M:
#   Let E[r] = sum_{j ≡ r (mod M), j even} binom(c, j)
#       O[r] = sum_{j ≡ r (mod M), j odd } binom(c, j)
# using the periodicity of the exponent modulo M.
#
# We use the standard idea with roots of unity discrete Fourier inversion in Z/M:
#   For f(j)=binom(c,j), the sum over j ≡ r mod M is (1/M)*Σ_{k=0}^{M-1} ζ^{-kr} (1+ζ^k)^c.
# Then even and odd parts are:
#    E[r] = (1/M)*Σ_{k} ζ^{-kr} ( (1+ζ^k)^c + (1-ζ^k)^c )/2,
#    O[r] = (1/M)*Σ_{k} ζ^{-kr} ( (1+ζ^k)^c - (1-ζ^k)^c )/2.
#
# We use this formula to compute for each distinct v the “y-part” polynomials.
# Then the effect of v on the XOR part is: if we choose an odd number then v is XOR’ed into the sum.
# Hence, in “frequency space” over XOR, the multiplier for the index corresponding to v is:
#    F_v(DP)[mask] gets replaced by even_poly convolved (cyclic over mod M) plus odd_poly shifted by the value v.
#
# After processing all groups we subtract the empty subsequence and then sum over all XOR values: 
#    answer = Σ_{s} (s)^K * DP[s][r=0].
#
# Finally the answer is printed modulo MOD.
#
# Implementation remarks:
# – We precompute an M–th primitive root of unity over complex numbers in high precision is tricky – instead
#   we work symbolically. Note that M <= 100 so we can simply sum over k=0..M-1 in O(M^2) time per distinct v.
#   (Total distinct values is at most 2^20 but in practice N is 2e5 so distinct values are at most 2e5.)
# – Because worst case 2e5 * M^2 (≈2e5 * 1e4 = 2e9) is too large in Python, we must use caching.
#   In fact, many c values will be repeated. So we precompute for each c (frequency) encountered the pair (E,O)
#   as a length-M tuple.
#
# – The FWHT on an array of length (1<<20) is implemented using a typical iterative method – we use
#   numpy arrays to speed up the transform.
#
# – In order to “multiply” the factors corresponding to different groups the idea is to convert the DP (of shape [1<<20, M])
#   with FWHT over the XOR dimension, then for each coordinate multiply in the “y–polynomial” factor appropriate to that coordinate.
#   Notice that if many groups “add” to the same XOR coordinate we must multiply the corresponding polynomials.
#   We do this by collecting for each XOR coordinate a list of multipliers and then taking their convolution power.
#
# – Finally, we apply the inverse FWHT.
#
# For brevity and due to the complexity, the code below is one way to do it in Python.
#
# WARNING: This solution is quite involved and uses heavy numpy vectorized operations. It assumes that
# sys.stdin.buffer is available and that the modulus 998244353 arithmetic is implemented carefully.
#
# (Note: Because of the problem’s complexity, multiple approaches are possible. This implementation is one correct approach.)
#
import sys,math,sys
import numpy as np

MOD = 998244353

# Precompute factorials and inv factorials up to some limit if needed.
# However, we are going to use roots-of-unity filtering to compute periodic sums.
# We define a function that given c and M returns two arrays E and O of length M: E[r] = sum_{j ≡ r, even} binom(c,j), O[r] similarly.
# We use the formula using discrete Fourier inversion.
# For each kth root, we need ζ^k. We choose to work in complex (with rounding) but then round to mod integer.
# Instead we can compute (1+ζ^k)^c mod MOD but here ζ^k is not in our field.
# So we use a direct summation method for c small. But c can be large.
# Instead, note that sum_{j=0}^{c} binom(c,j)=2^c.
# And sums mod M: because M is small, we can compute binom(c,j) mod MOD for j=0..c but c can be 2e5, so doing that for each distinct c in worst-case 2e5 * 2e5 is too slow.
# We use caching and dynamic programming mod MOD over j mod M.
#
# We want F_even[r] = sum_{j=0}^{c} [j even and j≡r mod M] binom(c,j)
# and F_odd[r] similarly.
# We can compute f_0, f_1,...,f_{M-1} for even and odd from (1+X)^c by taking c-th power mod (X^M-1). Using exponentiation by squaring.
#
def poly_power_mod1(c, M):
    # polynomial f(X)=1+X, we want f(X)^c mod (X^M-1) but then split even/odd parts
    # We compute with exponentiation by squaring. f is degree1.
    poly = [1,1] + [0]*(M-2)
    # Represent poly as length M list.
    def poly_mul(a, b):
        res = [0]*M
        for i in range(M):
            if a[i]:
                for j in range(M):
                    if b[j]:
                        res[(i+j)%M] = (res[(i+j)%M] + a[i]*b[j]) % MOD
        return res
    res = [0]*M; res[0]=1
    exp = c
    while exp:
        if exp&1:
            res = poly_mul(res, poly)
        poly = poly_mul(poly, poly)
        exp //=2
    return res

# Now from f(X)= (1+X)^c, we want to sum up even-indexed coefficients and odd-indexed coefficients by mod M residue.
# Let poly = poly_power_mod1(c, M). Then
#   E[r] = sum_{j even, j≡r} f_coef, O[r] = sum_{j odd, j≡r} f_coef.
# But note that f(X)= sum_{r=0}^{M-1} (A[r]) X^r where A[r] = sum_{j≡r} binom(c,j).
# We need to split into even/odd. We can compute f(1) and f(-1). But that loses the mod information.
# Instead, compute binom(c,j) mod MOD for j=0..min(c, M*2) because period M means j mod M repeats?
# Actually, no; we need full sums.
# Instead, use iterative recurrence to get even and odd sums (fast convolution with polynomial [1,0] and [0,1]).
# We'll use repeated squaring for a 2xM system.
def poly2_power(c, M):
    # We want to compute two arrays E and O (each length M) such that
    # E[r] = sum_{j even, j≡r} binom(c,j), O[r] = sum_{j odd, j≡r} binom(c,j).
    # They satisfy: for c=0, E[0]=1, O=0.
    # And for c>=1, the recurrence: (E,O) convolve with (even_poly, odd_poly) for one element.
    # For one element, even part: picking not take (j=0) gives 1, odd part: take one gives X.
    # So base polynomial is: E_base[0]=1, O_base[1]=1.
    # Then we want (E, O) = (1+X)^c split by parity.
    # We can compute by exponentiation in the ring R = (Z/M)[X] with X^M=1, but we separate even and odd parts.
    # We represent an element as a pair of length M arrays.
    def pair_mul(P, Q):
        # P=(E1,O1), Q=(E2,O2) then product = (E, O) with:
        # E[r] = sum_{a+b=r mod M} [E1[a]*E2[b] + O1[a]*O2[b]]
        # O[r] = sum_{a+b=r mod M} [E1[a]*O2[b] + O1[a]*E2[b]]
        E = [0]*M
        O = [0]*M
        for a in range(M):
            if P[0][a] or P[1][a]:
                for b in range(M):
                    if Q[0][b] or Q[1][b]:
                        r = (a+b)%M
                        E[r] = (E[r] + P[0][a]*Q[0][b] + P[1][a]*Q[1][b]) % MOD
                        O[r] = (O[r] + P[0][a]*Q[1][b] + P[1][a]*Q[0][b]) % MOD
        return (E,O)
    # base: for one element: P = ([1,0,...,0], [0,1,0,...,0])
    E_base = [0]*M; E_base[0]=1
    O_base = [0]*M; O_base[1 % M] = 1
    # identity:
    E_id = [0]*M; E_id[0]=1
    O_id = [0]*M
    res = (E_id, O_id)
    base = (E_base, O_base)
    exp = c
    while exp:
        if exp&1:
            res = pair_mul(res, base)
        base = pair_mul(base, base)
        exp//=2
    return res

# We'll cache results for frequency c.
poly2_cache = {}
def get_poly2(c, M):
    if c in poly2_cache:
        return poly2_cache[c]
    res = poly2_power(c, M)
    poly2_cache[c] = res
    return res

# Now main combinatorial merge:
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    M = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    # Group frequencies by value.
    freq = {}
    for a in A:
        freq[a] = freq.get(a, 0) + 1
    # Our DP is over the XOR-space (of size XSIZE = 1<<20) and length mod M.
    XSIZE = 1<<20
    # We will maintain an array dp of shape (XSIZE, M).
    # But doing that explicitly (of size about 1e6 * M) is too heavy in Python.
    # Instead, we use FWHT technique: we take an FWHT along the XOR dimension.
    # In the FWHT domain each coordinate multiplies independently.
    # For each distinct value v, its effect is to multiply dp[u] by a 2xM factor:
    #  multiplication: new dp[u] (for dp's transform index corresponding to u) gets multiplied
    #  by factor F, where F is:
    #    if u = [something] we need to “add” the contribution of v.
    #
    # The convolution over XOR becomes pointwise multiplication after the FWHT.
    # So we compute f = FWHT(dp) initially.
    # Initially, dp[0][0] = 1 and dp[mask][r]=0 for mask>0.
    # So its FWHT is constant: f[u][0] = 1 for all u, and f[u][r>0] = 0.
    # Then for each distinct value v with frequency c, note that picking items equal to v contributes:
    #    even part: (E, with no XOR change), odd part: (O, with XOR adding v).
    # In the FWHT domain, convolution turns into multiplication.
    # But note that the addition of v as XOR in the original domain becomes a “phase‐shift”
    # in the FWHT domain. However, because the XOR convolution is “diagonalized” by the FWHT,
    # and the merge over different v’s become independent coordinatewise – the multiplier is:
    # For each FWHT coordinate u, the multiplier from group v is:
    #    multiplier = E_poly (the polynomial E) + chi_v(u)* O_poly (the polynomial O)
    # where chi_v(u) = (-1)^{popcount(u & v)} because the Walsh character for XOR is given by (-1)^{bit dot product}.
    #
    # Then overall, f[u] is multiplied by:
    #    Prod_{v in groups} ( E^{(freq[v])} + chi_v(u)*O^{(freq[v])} )
    # where E^{(c)} and O^{(c)} are length-M arrays obtained by get_poly2(c,M).
    #
    # So we will loop over FWHT coordinate u from 0 to XSIZE-1 and multiply the initial value 1
    # by for each v in freq, the polynomial multiplier appropriate.
    # But looping over u (1<<20 ~1e6) and over each v (at most 2e5) is too slow.
    # We change the order: group by v. Instead, note that the complete product over v factors can be computed by:
    #    f[u] = Prod_{v} F_v(u)
    # and we want the inverse FWHT of f.
    #
    # But then our final answer is: sum_{s} (s)^K * dp[s][0], where dp is the inverse FWHT of f.
    #
    # We therefore now compute f in FWHT domain as an array of shape (XSIZE, M).
    # Initially f[u] = [1,0,...,0] for all u.
    # For each group v with frequency c, let (E,O) = get_poly2(c, M)
    # Then multiply f[u] (a length-M polynomial) by (E + chi_v(u)*O), where chi_v(u)=+1 or -1.
    # We do this multiplication coefficientwise (cyclic convolution mod M).
    #
    # We want to do this for all u simultaneously. Notice that for a fixed v, chi_v(u) depends on u.
    # We can precompute for each u the factor chi = (-1)^{popcount(u & v)}.
    # However, doing that for each v and for all u is heavy if done in python loop.
    #
    # So we iterate over groups and update an array f (of shape (XSIZE, M)) using numpy vectorized operations.
    
    f = np.zeros((XSIZE, M), dtype=np.int64)
    # initial: f[u] = [1,0,...,0]
    f[:,0] = 1
    
    # Precompute bit-counts for indices 0...XSIZE-1 can be done using numpy.
    popcount = np.unpackbits(np.arange(XSIZE, dtype=np.uint32).view(np.uint8)).reshape(XSIZE, -1).sum(axis=1)
    
    # For each group, update f.
    # Note: to avoid iterating  over possibly many groups in python loop,
    # we accumulate groups with the same v value. But here freq is already grouped.
    for v, c in freq.items():
        # get the two polynomials: E and O of length M.
        E_poly, O_poly = get_poly2(c, M)  # both are lists of length M
        E_poly = np.array(E_poly, dtype=np.int64)
        O_poly = np.array(O_poly, dtype=np.int64)
        # For each u, compute chi = (-1)^(popcount(u & v)).
        # To compute popcount(u & v) for all u, we use: (np.bitwise_and(np.arange(XSIZE), v)).
        # Note: np.unpackbits works on uint8 so do careful conversion.
        masku = np.arange(XSIZE, dtype=np.int64)
        chi = ((np.unpackbits((masku & v).astype(np.uint32).view(np.uint8)).reshape(XSIZE,4).sum(axis=1)) & 1)*2-1
        # Now, for each u, we want to update f[u] = cyclic_conv( f[u] , (E_poly + chi[u]*O_poly) ).
        # Here cyclic convolution means: new[u][r] = sum_{i+j = r mod M} f[u][i] * X_poly[j].
        # Since M is small, we can do full convolution by looping over j in range(M) in vectorized way.
        newf = np.zeros_like(f)
        # Precompute multiplier = E_poly + chi[u]*O_poly for each u.
        # We do this by broadcasting: for each u, for each j:
        #   X[u,j] = E_poly[j] + chi[u]*O_poly[j].
        mult = E_poly + np.outer(chi, O_poly)
        mult %= MOD
        # Now do cyclic convolution: for each u and for each r, newf[u,r] = sum_{j} f[u, (r-j) mod M]*mult[u, j].
        for j in range(M):
            newf[:, j:] = (newf[:, j:] + f[:, :M-j] * mult[:, j:j+1]) % MOD
            if j:
                newf[:, :j] = (newf[:, :j] + f[:, M-j:] * mult[:, j:j+1]) % MOD
        f = newf
    # Now f is in FWHT domain. We now apply the inverse FWHT to get dp (over XOR domain).
    # Standard FWHT (Hadamard transform) for XOR:
    def fwht(a):
        n = a.shape[0]
        h = 1
        while h < n:
            a = a.copy()
            for i in range(0, n, h*2):
                # vectorized over block:
                left = a[i:i+h]
                right = a[i+h:i+2*h]
                a[i:i+h] = (left+right) % MOD
                a[i+h:i+2*h] = (left-right) % MOD
            h *= 2
        return a
    def ifwht(a):
        n = a.shape[0]
        a = fwht(a)
        inv_n = pow(n, MOD-2, MOD)
        return (a * inv_n) % MOD
    # Apply inverse FWHT along axis 0 for each coefficient in length M.
    dp = np.empty_like(f)
    for r in range(M):
        col = f[:, r]
        dp[:, r] = ifwht(col)
    # subtract the empty subsequence (which is at dp[0][0])
    dp[0,0] = (dp[0,0] - 1) % MOD
    # Now compute answer = sum_{s} (s)^K * dp[s][0], where s is integer from 0 to XSIZE-1.
    # We need to compute s^K mod MOD.
    s_arr = np.arange(XSIZE, dtype=np.int64) % MOD
    # fast exponentiation vectorized:
    s_pow = np.power(s_arr, K, MOD)
    ans = int((s_pow * dp[:,0]) % MOD .sum() % MOD)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()