#!/usr/bin/env python3
mod = 998244353

# ---- modular inverse and binomial coefficients ----------------
def modinv(a, mod=mod):
    return pow(a, mod-2, mod)

# Precompute binomials for small n (we only need up to ~50 here)
MAXN = 50
binom = [[0]*(MAXN+1) for _ in range(MAXN+1)]
for i in range(MAXN+1):
    binom[i][0] = 1
    for j in range(1, i+1):
        binom[i][j] = (binom[i-1][j-1] + binom[i-1][j]) % mod

# ---- poly_geosum: recursively compute S[d] = sum_{n=0}^{N} n^d * r^n for d = 0...D,
# and return also P = r^(N+1)
def poly_geosum_helper(r, N, D):
    if N < 0:
        return ([0]*(D+1), 1)
    if N == 0:
        S = [0]*(D+1)
        S[0] = 1  # only n=0 term: 0^0 is 1.
        P = r % mod
        return (S, P)
    m = (N+1)//2
    S_left, P_left = poly_geosum_helper(r, m-1, D)
    S_right, P_right = poly_geosum_helper(r, N-m, D)
    pow_r_m = pow(r, m, mod)
    S_total = [0]*(D+1)
    for j in range(D+1):
        S_total[j] = S_left[j] % mod
        add_val = 0
        for i in range(j+1):
            add_val = (add_val + binom[j][i] * pow(m, j-i, mod) * S_right[i]) % mod
        S_total[j] = (S_total[j] + pow_r_m * add_val) % mod
    P_total = (pow_r_m * P_right) % mod
    return (S_total, P_total)
 
def poly_geosum(r, N, D):
    S, _ = poly_geosum_helper(r, N, D)
    return S

# ---- multivariate polynomial multiplication ------------------
# We represent a monomial in K variables by a tuple of K nonnegative ints.
def add_exponents(v, w):
    return tuple(v_i + w_i for v_i, w_i in zip(v, w))
 
def poly_mul(polyA, polyB, K):
    res = {}
    for monA, coefA in polyA.items():
        for monB, coefB in polyB.items():
            mon = add_exponents(monA, monB)
            res[mon] = (res.get(mon, 0) + coefA * coefB) % mod
    return res

# ---- Build the “base–polynomial” Q(z) = ∑ₓ z^(v(x))
def valuation(x, p):
    cnt = 0
    while x % p == 0:
        cnt += 1
        x //= p
    return cnt
 
def build_Q(M_val, primes):
    Q = {}
    for x in range(1, M_val+1):
        vec = []
        for p in primes:
            cnt = 0
            temp = x
            while temp % p == 0:
                cnt += 1
                temp //= p
            vec.append(cnt)
        vec = tuple(vec)
        Q[vec] = (Q.get(vec,0) + 1) % mod
    return Q

# ---- “Evaluate” the polynomial at z_i=1 after applying the (1+d/dz) operators.
# (For a monomial z^(v) the “weight” is ∏ (v_i+1).
def evaluate_poly(poly):
    total = 0
    for mon, coef in poly.items():
        prod = 1
        for exp in mon:
            prod = (prod * (exp+1)) % mod
        total = (total + coef * prod) % mod
    return total

# ---- Compute Q^L (by iterative convolution)
def poly_power(Q, L, K):
    res = None
    for _ in range(L):
        if res is None:
            res = Q.copy()
        else:
            res = poly_mul(res, Q, K)
    return res

# ---- Lagrange interpolation in one variable.
# Given points (xs[i], ys[i]) for i = 0,..., n-1, return a list of coefficients
# of the unique polynomial (degree < n) P(x) = sum_{j} c_j x^j.
def lagrange_interpolation(xs, ys):
    n = len(xs)
    basis = []
    for i in range(n):
        poly_i = [1]
        denom = 1
        for j in range(n):
            if i == j: continue
            new_poly = [0]*(len(poly_i)+1)
            for k in range(len(poly_i)):
                new_poly[k+1] = (new_poly[k+1] + poly_i[k]) % mod
                new_poly[k] = (new_poly[k] - xs[j]*poly_i[k]) % mod
            poly_i = new_poly
            denom = (denom * (xs[i]-xs[j])) % mod
        inv_denom = modinv(denom, mod)
        for k in range(len(poly_i)):
            poly_i[k] = (poly_i[k] * inv_denom) % mod
        basis.append(poly_i)
    res_poly = [0]*n
    for i in range(n):
        for k in range(len(basis[i])):
            res_poly[k] = (res_poly[k] + ys[i] * basis[i][k]) % mod
    return res_poly

# ---- Main
def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M_val = int(data[1])
    # If M==1 then every sequence equals 1 and d(1)=1 so answer = N mod mod.
    if M_val == 1:
        print(N % mod)
        return
    # Build list of primes p ≤ M that appear in some x ∈ {1,…,M}.
    basePrimes = [2,3,5,7,11,13]
    primes = []
    for p in basePrimes:
        if p <= M_val:
            for x in range(1, M_val+1):
                if x % p == 0:
                    primes.append(p)
                    break
    K = len(primes)
    # (If K==0 then M==1, but we already dealt with that.)
 
    # For our later interpolation we will compute f(L) for L = 1,..,L_max.
    # (It turns out that one may prove f(L) = M^L * P(L) where P is a polynomial
    # of degree at most K; so we compute for L=1,...,K+1.)
    L_max = K+1
    if N < L_max:
        L_max = N
    Qpoly = build_Q(M_val, primes)
    fvals = []  # store f(1), f(2), …, f(L_max)
    for L in range(1, L_max+1):
        polyL = poly_power(Qpoly, L, K)
        fL = evaluate_poly(polyL) % mod
        fvals.append(fL)
    # f(L) = M^L * P(L) so let
    Ppoints = []
    for i,fL in enumerate(fvals, start=1):
        invM = modinv(pow(M_val, i, mod), mod)
        Ppoints.append((fL * invM) % mod)
    xs = list(range(1, L_max+1))
    # Interpolate polynomial P(x) of degree < L_max with P(i)=Ppoints[i-1]
    polyCoeffs = lagrange_interpolation(xs, Ppoints)
    d = L_max - 1  # degree of our polynomial (at most)
 
    # Then f(L)=M^L * (c0 + c1*L + … + c_d*L^d).
    # Our desired answer is S = sum_{L=1}^{N} f(L) = sum_{L=1}^N M^L * P(L).
    # Swap the summation:
    #     S = sum_{j=0}^d c_j * (sum_{L=1}^N L^j * M^L).
    # We now use our poly_geosum routine to compute T(j) = sum_{L=0}^{N} L^j * M^L.
    Tvals = []
    for j in range(d+1):
        S_poly = poly_geosum(M_val, N, j)
        # For j==0 the L=0 term is 1 and we want L>=1 so subtract 1.
        if j == 0:
            Tvals.append((S_poly[0]-1) % mod)
        else:
            Tvals.append(S_poly[j] % mod)
    S_total = 0
    for j in range(d+1):
        S_total = (S_total + polyCoeffs[j] * Tvals[j]) % mod
    print(S_total % mod)
 
if __name__ == '__main__':
    main()