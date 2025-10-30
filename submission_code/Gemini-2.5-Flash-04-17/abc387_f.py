import sys

def power(base, exp):
    res = 1
    base %= 998244353
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % 998244353
        base = (base * base) % 998244353
        exp //= 2
    return res

def mod_inverse(n):
    n = (n % 998244353 + 998244353) % 998244353
    return power(n, 998244353 - 2)


def sum_of_powers(M, k):
    MOD = 998244353

    if k == 0:
        # Sum V^0 = Sum 1 for V=1 to M is M
        return M % MOD

    # Compute S_k(m) = sum_{i=1}^m i^k for m = 0 to k+1
    # The sum of k-th powers is a polynomial in m of degree k+1.
    # We need k+2 points to uniquely determine this polynomial.
    num_points = k + 2
    
    S = [0] * num_points
    for m in range(1, num_points):
        S[m] = (S[m-1] + power(m, k)) % MOD

    # If M is small, we can just return the precomputed value
    if M < num_points: # M is 1-indexed, points are 0..k+1 (k+2 values)
        return S[M]

    # Lagrange Interpolation
    # S_k(M) = sum_{j=0}^{k+1} S_k(j) * prod_{i=0, i!=j}^{k+1} (M-i)/(j-i)
    
    ans = 0
    for j in range(num_points):
        # If S[j] is 0, this term is 0, skip calculation
        if S[j] == 0:
            continue
            
        term = S[j]
        
        # Compute numerator prod (M-i) for i = 0 .. k+1, i != j
        num = 1
        for i in range(num_points):
            if i != j:
                num = (num * (M - i)) % MOD
        
        # Compute denominator prod (j-i) for i = 0 .. k+1, i != j
        den = 1
        for i in range(num_points):
            if i != j:
                den = (den * (j - i)) % MOD

        # Need modular inverse of den
        den_inv = mod_inverse(den)
        
        term = (term * num) % MOD
        term = (term * den_inv) % MOD
        
        ans = (ans + term) % MOD
    
    # Ensure positive result after modulo
    ans = (ans + MOD) % MOD
    return ans


def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate k = N - |Im(A)| = number of nodes in {1..N} not in the image of A
    # These are the nodes that are not pointed to by any other node.
    image_set = set(A)
    k = N - len(image_set)

    # The answer is sum_{V=1}^M V^k mod 998244353
    result = sum_of_powers(M, k)
    print(result)

# Redirect stdin and stdout for submission format
if __name__ == "__main__":
    # Standard input/output is handled directly by the code
    # The solve() function reads from stdin and writes to stdout
    solve()