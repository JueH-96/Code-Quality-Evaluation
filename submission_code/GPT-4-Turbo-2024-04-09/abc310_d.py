from itertools import product
from collections import defaultdict

def count_valid_teams(N, T, M, incompatibilities):
    # Edge case: if T == 1, all players must be in one team, check if there are any incompatibilities
    if T == 1:
        return 0 if M > 0 else 1
    
    # Edge case: if there are no incompatibilities, we just need to count the ways to partition N players into T teams
    if M == 0:
        # Using Stirling numbers of the second kind to count partitions of N items into T non-empty sets
        # S(n, k) = k * S(n-1, k) + S(n-1, k-1)
        # Base cases: S(n, 1) = 1, S(n, n) = 1, S(n, k) = 0 for k > n
        stirling = [[0] * (N + 1) for _ in range(N + 1)]
        for n in range(1, N + 1):
            stirling[n][1] = 1
            stirling[n][n] = 1
            for k in range(2, n):
                stirling[n][k] = k * stirling[n-1][k] + stirling[n-1][k-1]
        
        return stirling[N][T]
    
    # Convert 1-based index to 0-based for easier array manipulation
    incompatibilities = [(a-1, b-1) for a, b in incompatibilities]
    
    # Total number of ways to assign players to T teams
    total_ways = 0
    
    # Iterate over all possible team assignments using T colors
    for assignment in product(range(T), repeat=N):
        # Check if this assignment is valid
        valid = True
        for a, b in incompatibilities:
            if assignment[a] == assignment[b]:
                valid = False
                break
        
        if valid:
            # Check if all teams are non-empty
            if len(set(assignment)) == T:
                total_ways += 1
    
    return total_ways

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
M = int(data[2])

incompatibilities = []
index = 3
for _ in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    incompatibilities.append((A, B))
    index += 2

result = count_valid_teams(N, T, M, incompatibilities)
print(result)