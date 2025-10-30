# We'll use a generating function approach.
# For each letter with allowed count up to C (or up to K, since only strings of length ≤ K are considered),
# its contribution is:
#
#      P_i(x) = 1 + x/1! + x^2/2! + ... + x^(min(C_i,K))/((min(C_i,K))!)
#
# Then the generating function for the whole nameplate is:
#
#      F(x) = ∏_{i=1}^{26} P_i(x)
#
# The coefficient of x^L in F(x) is
#      a_L = sum (over f_1+...+f_26 = L with f_i <= C_i) 1/(f_1! * ... * f_26!)
#
# Multiplying this coefficient by L! gives the number of strings of length L
# satisfying the restrictions:
#
#      count(L) = L! * a_L = sum_{f_i} (L!/(f_1! ... f_26!))
#
# We want the total over L = 1 to K.
#
# Because K and all C_i are at most 1000 and there are only 26 letters,
# we can compute the product polynomial F(x) mod mod, only keeping x^j for j = 0..K.
#
# Since the terms in P_i(x) are 1/j! (mod mod), we first precompute
# factorials and inv-factorials up to K.
#
# We then represent polynomials as numpy arrays of length (K+1) (coefficients for x^0 to x^K)
# and perform convolution for each letter factor. To avoid Python-level O((K^2)*26) loops,
# we write a convolution routine that loops over the current polynomial (at most 1001 coefficients)
# and uses numpy vectorized operations on each slice (which is efficient in C).
#
# Finally, we sum over L from 1 to K of fact[L] * poly[L] modulo 998244353.
#
# Time complexity: roughly 26*(K+1)*O(L) with L up to K, so around 26e6 operations in worst case (performed in C code via numpy),
# which is acceptable.
#
# The complete solution is given below.
 
def main():
    import sys
    import numpy as np
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    K = int(data[0])
    # Read 26 integers for the maximum allowed occurrences for A through Z.
    C = list(map(int, data[1:1+26]))
     
    # Precompute factorials and inverse factorials up to K.
    fact = [1] * (K+1)
    invfact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % mod
    # Compute inverse factorial for K using Fermat's little theorem,
    # then fill in downward.
    invfact[K] = pow(fact[K], mod - 2, mod)
    for i in range(K, 0, -1):
        invfact[i-1] = invfact[i] * i % mod
 
    # Start with the polynomial "1" (i.e. coefficient 1 for x^0).
    poly = np.zeros(K+1, dtype=np.int64)
    poly[0] = 1
 
    # Define convolution under mod, truncated to degree K.
    # Given two polynomials A and B, represented as numpy arrays of length up to K+1,
    # this computes C(x) = A(x)*B(x) with only terms x^0, x^1, ..., x^K.
    def poly_conv(A, B):
        la = len(A)
        lb = len(B)
        res = np.zeros(K+1, dtype=np.int64)
        for i in range(la):
            val = A[i]
            if val == 0:
                continue
            # For fixed i in "A", B contributes to degrees i, i+1, ...,
            # but we can only add up to degree K.
            r = lb if i + lb <= K + 1 else (K + 1 - i)
            res[i:i+r] = (res[i:i+r] + val * B[:r]) % mod
        return res
 
    # Multiply the polynomial for each letter.
    # For letter i, allowed count is from 0 to min(C[i], K)
    # and its generating function is sum_{j=0}^{min(C_i,K)} invfact[j] * x^j.
    for i in range(26):
        bound = C[i] if C[i] < K else K
        poly_letter = np.array(invfact[:bound+1], dtype=np.int64)
        poly = poly_conv(poly, poly_letter)
 
    # The coefficient poly[L] corresponds to a_L in F(x).
    # Multiplying by fact[L] gives the number of strings of length L.
    ans = 0
    for L in range(1, K+1):
        ans = (ans + fact[L] * int(poly[L])) % mod
    sys.stdout.write(str(ans))
 
if __name__ == '__main__':
    main()