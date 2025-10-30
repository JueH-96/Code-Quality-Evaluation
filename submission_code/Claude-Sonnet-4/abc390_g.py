def solve():
    MOD = 998244353
    N = int(input())
    
    # Precompute factorials
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    # Precompute powers of 10
    max_digits = 6 * N  # Conservative upper bound
    pow10 = [1] * (max_digits + 1)
    for i in range(1, max_digits + 1):
        pow10[i] = (pow10[i-1] * 10) % MOD
    
    result = 0
    
    # For each number i from 1 to N
    for i in range(1, N + 1):
        # For each position pos where i can appear (0-indexed)
        for pos in range(N):
            # Number of times i appears at position pos
            freq = fact[N - 1]
            
            # Calculate total digits that come after position pos
            # This is the sum of digits of all numbers except i
            total_digits_after = sum(len(str(j)) for j in range(1, N + 1) if j != i)
            
            # Contribution of number i at position pos
            contribution = (i * pow10[total_digits_after] % MOD * freq) % MOD
            result = (result + contribution) % MOD
    
    print(result)

solve()