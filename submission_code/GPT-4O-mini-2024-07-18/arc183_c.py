def count_valid_permutations(N, M, conditions):
    MOD = 998244353

    # Initialize a dp array where dp[i] is the number of valid permutations of size i
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to arrange zero elements

    # Create a list to store the constraints
    constraints = [[] for _ in range(N + 1)]
    
    for L, R, X in conditions:
        constraints[R].append((L, X))

    # Iterate through all possible sizes of permutations
    for size in range(1, N + 1):
        # Start with the number of ways to arrange `size` elements
        dp[size] = dp[size - 1] * size % MOD
        
        # Apply constraints for the current size
        for L, X in constraints[size]:
            # If we have a constraint that the maximum in [L, size] is not P[X]
            # We need to subtract the invalid arrangements
            if L <= size and X <= size:
                # Calculate the number of invalid arrangements
                # The number of ways to arrange the remaining elements
                remaining = size - 1  # Exclude P[X]
                invalid_count = dp[remaining] * (remaining + 1) % MOD
                dp[size] = (dp[size] - invalid_count + MOD) % MOD

    return dp[N]

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, M = map(int, data[0].split())
    conditions = [tuple(map(int, line.split())) for line in data[1:M + 1]]
    
    result = count_valid_permutations(N, M, conditions)
    print(result)

if __name__ == "__main__":
    main()