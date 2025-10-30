def solve(N, M):
    MOD = 998244353
    
    # Calculate (M-1)^N mod MOD
    power = pow(M-1, N, MOD)
    
    # Calculate (-1)^N * (M-1) mod MOD
    # When N is even, (-1)^N = 1
    # When N is odd, (-1)^N = -1
    if N % 2 == 0:
        term = (M-1) % MOD
    else:
        term = (MOD - (M-1)) % MOD  # This handles the negative value in modular arithmetic
    
    # Combine the terms
    result = (power + term) % MOD
    
    return result

# Read input
N, M = map(int, input().split())
print(solve(N, M))