# YOUR CODE HERE
MOD = 998244353

def pow_mod(x, y, m):
    result = 1
    x = x % m
    while y > 0:
        if y % 2 == 1:
            result = (result * x) % m
        y = y >> 1
        x = (x * x) % m
    return result

def solve():
    import sys
    N = int(sys.stdin.read().strip())
    s = str(N)
    len_s = len(s)
    num = int(s)
    
    # Sum of geometric series: num * (1 + 10^len_s + 10^(2*len_s) + ... + 10^((N-1)*len_s))
    # This can be written as num * (10^(N*len_s) - 1) / (10^len_s - 1)
    # We need to compute this modulo 998244353
    
    # First, compute 10^(N*len_s) % MOD
    power = pow_mod(10, N * len_s, MOD)
    
    # Compute (10^(N*len_s) - 1) % MOD
    numerator = (power - 1) % MOD
    
    # Compute (10^len_s - 1) % MOD
    denominator = (pow_mod(10, len_s, MOD) - 1) % MOD
    
    # Compute modular inverse of denominator
    # Using Fermat's little theorem: a^(p-1) ≡ 1 (mod p) => a^(p-2) ≡ a^(-1) (mod p)
    inverse_denominator = pow_mod(denominator, MOD - 2, MOD)
    
    # Compute the result
    result = (num * numerator * inverse_denominator) % MOD
    
    print(result)

solve()