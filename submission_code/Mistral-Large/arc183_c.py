import sys
from math import factorial
from collections import defaultdict

MOD = 998244353

def count_valid_permutations(N, conditions):
    # Initialize a DP table to store the number of valid permutations
    dp = [0] * (1 << N)
    dp[0] = 1

    # Initialize a list to store the maximum value for each subset
    max_value = [0] * (1 << N)

    # Precompute the maximum value for each subset
    for mask in range(1, 1 << N):
        for i in range(N):
            if mask & (1 << i):
                max_value[mask] = max(max_value[mask], i + 1)

    # Process each condition
    for L, R, X in conditions:
        X -= 1  # Convert to 0-based index
        for mask in range(1 << N):
            if dp[mask] > 0:
                subset = mask & ((1 << (R - L + 1)) - 1) << L
                if max_value[subset] != X + 1:
                    dp[mask | (1 << X)] = (dp[mask | (1 << X)] + dp[mask]) % MOD

    # The result is the sum of all valid permutations
    result = sum(dp) % MOD
    return result

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    conditions = []

    index = 2
    for _ in range(M):
        L = int(data[index])
        R = int(data[index + 1])
        X = int(data[index + 2])
        conditions.append((L, R, X))
        index += 3

    result = count_valid_permutations(N, conditions)
    print(result)

if __name__ == "__main__":
    main()