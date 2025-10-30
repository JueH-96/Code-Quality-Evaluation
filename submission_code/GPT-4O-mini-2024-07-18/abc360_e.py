def mod_inverse(a, p):
    return pow(a, p - 2, p)

def expected_position(N, K):
    MOD = 998244353
    
    # The expected position of the black ball after K operations
    # E[x] = (1 + 2 + ... + N) / N = (N * (N + 1) / 2) / N = (N + 1) / 2
    # After K operations, the expected position becomes:
    # E[x] = (N + 1) / 2 * (1 - (1 - 1/N)^(2K))
    
    # Calculate (N + 1) / 2
    numerator = (N + 1) % MOD
    denominator = 2 % MOD
    
    # Calculate (1 - 1/N)^(2K)
    one_minus_inv_n = (N - 1) * mod_inverse(N, MOD) % MOD
    term = pow(one_minus_inv_n, 2 * K, MOD)
    
    # Expected value
    expected_value = (numerator * (1 - term) % MOD) * mod_inverse(1 - (1 - 1/N) ** (2 * K), MOD) % MOD
    
    # We need to find R such that R * Q ≡ P (mod 998244353)
    # Here P = expected_value and Q = 1 (since we are looking for the expected position)
    R = expected_value % MOD
    
    return R

import sys
input = sys.stdin.read

def main():
    N, K = map(int, input().strip().split())
    result = expected_position(N, K)
    print(result)

if __name__ == "__main__":
    main()