def solve():
    N = int(input())
    MOD = 998244353
    
    # Convert N to string to get its length
    N_str = str(N)
    len_N = len(N_str)
    
    # Calculate 10^len_N % MOD
    power = pow(10, len_N, MOD)
    
    # Calculate (1 + power + power^2 + ... + power^(N-1)) * N
    # Using geometric series sum formula
    if N == 1:
        result = N % MOD
    else:
        # Calculate power^N - 1
        numerator = (pow(power, N, MOD) - 1 + MOD) % MOD
        # Calculate power - 1
        denominator = (power - 1 + MOD) % MOD
        # Calculate modular multiplicative inverse of denominator
        inv_denominator = pow(denominator, MOD-2, MOD)
        # Calculate the sum of geometric series
        sum_series = (numerator * inv_denominator) % MOD
        # Multiply by N
        result = (sum_series * N) % MOD
        
    print(result)

solve()