def solve():
    A, B, M = map(int, input().split())
    
    # We need to count permutations of length AB-1 with LIS=A and LDS=B
    # that satisfy the additional condition
    
    # Using dynamic programming approach
    # dp[i][j] = number of ways to fill positions considering constraints
    
    # The answer is related to counting SYT of shape (B,B,...,B,B-1)
    # where we have A-1 rows of length B and 1 row of length B-1
    
    # For the specific case, we use the hook length formula
    # But we need to account for the special condition
    
    # After analysis, the count is:
    # C(A+B-2, A-1) * C(A+B-2, B-1) * (AB-1)! / (A*B)
    
    # Calculate factorials and inverse factorials modulo M
    max_n = A + B + AB
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % M
    
    # Calculate modular inverse using Fermat's little theorem
    def modinv(x, m):
        return pow(x, m - 2, m)
    
    # Calculate C(n, k) mod M
    def comb(n, k, m):
        if k > n or k < 0:
            return 0
        return fact[n] * modinv(fact[k], m) % m * modinv(fact[n-k], m) % m
    
    # The formula for our specific problem
    result = comb(A + B - 2, A - 1, M) * comb(A + B - 2, B - 1, M) % M
    result = result * fact[AB - 1] % M
    result = result * modinv(A * B % M, M) % M
    
    print(result)

solve()