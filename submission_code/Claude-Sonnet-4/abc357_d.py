# YOUR CODE HERE
def solve():
    MOD = 998244353
    N = int(input())
    
    # Calculate number of digits in N
    d = len(str(N))
    
    # We need to calculate N * (10^(N*d) - 1) / (10^d - 1) mod MOD
    
    # First calculate 10^d mod MOD
    base = pow(10, d, MOD)
    
    # Calculate 10^(N*d) mod MOD using fast exponentiation
    # Since N*d can be very large, we use Fermat's little theorem
    # 10^(N*d) ≡ 10^((N*d) mod (MOD-1)) (mod MOD)
    
    # Calculate (N*d) mod (MOD-1)
    nd_mod = (N % (MOD - 1)) * (d % (MOD - 1)) % (MOD - 1)
    
    # Calculate 10^(N*d) mod MOD
    power_nd = pow(10, nd_mod, MOD)
    
    # Calculate numerator: N * (10^(N*d) - 1)
    numerator = (N % MOD) * ((power_nd - 1) % MOD) % MOD
    
    # Calculate denominator: 10^d - 1
    denominator = (base - 1) % MOD
    
    # Calculate modular inverse of denominator
    inv_denominator = pow(denominator, MOD - 2, MOD)
    
    # Final result
    result = (numerator * inv_denominator) % MOD
    
    print(result)

solve()