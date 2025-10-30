# YOUR CODE HERE
def solve(N, S):
    MOD = 10**9 + 7
    result = 0
    power_of_10 = 1
    
    for i in range(N - 1, -1, -1):
        digit = int(S[i])
        
        # Calculate contribution of this digit to all substrings ending at or after i
        result += digit * (i + 1) * power_of_10
        result %= MOD
        
        # Update power_of_10 for next iteration
        power_of_10 = (power_of_10 * 10 + 1) % MOD

    return result

# Read input
N = int(input())
S = input().strip()

# Solve and print result
print(solve(N, S))