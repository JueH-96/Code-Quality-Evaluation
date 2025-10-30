from collections import defaultdict
from itertools import permutations

MOD = 998244353

def solve():
    N, M = map(int, input().split())
    conditions = []
    for _ in range(M):
        L, R, X = map(int, input().split())
        conditions.append((L, R, X))

    # Create a dictionary to store the number of permutations that satisfy each condition
    dp = defaultdict(int)

    # Iterate over all possible permutations
    for perm in permutations(range(1, N+1)):
        is_valid = True
        for L, R, X in conditions:
            if max(perm[L-1:R]) == perm[X-1]:
                is_valid = False
                break
        if is_valid:
            key = tuple(perm)
            dp[key] += 1

    # Calculate the total number of valid permutations
    total = sum(dp.values()) % MOD
    return total

print(solve())