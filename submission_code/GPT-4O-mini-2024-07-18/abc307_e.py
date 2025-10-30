def count_ways(N, M):
    MOD = 998244353
    
    if M == 1:
        return 0 if N > 1 else 1
    
    # Calculate the number of valid distributions
    # Using the formula: M * (M - 1)^(N - 1) for circular arrangement
    result = M * pow(M - 1, N - 1, MOD) % MOD
    return result

# Read input
import sys
input = sys.stdin.read
N, M = map(int, input().strip().split())

# Output the result
print(count_ways(N, M))