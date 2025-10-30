from collections import defaultdict
from itertools import combinations

MOD = 998244353

def count_arithmetic_subsequences(N, A):
    # Initialize the result list with the count of subsequences of length 1
    result = [N]
    
    # Precompute the differences for all pairs
    diffs = defaultdict(int)
    for i in range(N):
        for j in range(i + 1, N):
            diffs[j - i, A[j] - A[i]] += 1
    
    # Count arithmetic subsequences of length >= 2
    for k in range(2, N + 1):
        count = 0
        for comb in combinations(range(N), k):
            valid = True
            diff = None
            for i in range(1, k):
                current_diff = A[comb[i]] - A[comb[i - 1]]
                if diff is None:
                    diff = current_diff
                elif diff != current_diff:
                    valid = False
                    break
            if valid:
                count += 1
        result.append(count % MOD)
    
    return result

# Read input
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Solve the problem
result = count_arithmetic_subsequences(N, A)

# Print the result
print(' '.join(map(str, result)))