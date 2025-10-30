def count_sequences(N, Q, operations):
    MOD = 998244353
    
    # Initialize the dp array
    dp = [[0] * (N + 1) for _ in range(Q + 1)]
    dp[0][0] = 1  # Base case: 1 way to do nothing
    
    # Process each operation
    for i in range(1, Q + 1):
        P_i, V_i = operations[i - 1]
        
        # We need to keep track of the maximum value we can set
        max_value = 0
        
        # Update dp for the current operation
        for j in range(i):
            # If we choose to replace S[1] to S[P_i]
            if P_i <= N:
                if max_value <= V_i:
                    dp[i][P_i] = (dp[i][P_i] + dp[j][P_i - 1]) % MOD
            
            # If we choose to replace S[P_i] to S[N]
            if P_i <= N:
                if max_value <= V_i:
                    dp[i][N] = (dp[i][N] + dp[j][N]) % MOD
        
        # Update the max_value for the next iteration
        max_value = max(max_value, V_i)
    
    # The answer is the sum of all valid sequences after Q operations
    result = sum(dp[Q]) % MOD
    return result

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, Q = map(int, data[0].split())
    operations = [tuple(map(int, line.split())) for line in data[1:]]
    
    result = count_sequences(N, Q, operations)
    print(result)

if __name__ == "__main__":
    main()