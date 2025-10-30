# YOUR CODE HERE
MOD = 998244353

def pow_mod(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def solve(N):
    # Calculate 10^len(N) % MOD
    len_N = len(str(N))
    ten_pow = pow_mod(10, len_N, MOD)
    
    # Calculate (10^len(N) - 1) % MOD
    denominator = (ten_pow - 1) % MOD
    
    # Calculate N * (10^(N*len(N)) - 1) / (10^len(N) - 1) % MOD
    numerator = (N * (pow_mod(ten_pow, N, MOD) - 1)) % MOD
    
    # Use Fermat's little theorem for modular division
    result = (numerator * pow_mod(denominator, MOD - 2, MOD)) % MOD
    
    return result

N = int(input())
print(solve(N))