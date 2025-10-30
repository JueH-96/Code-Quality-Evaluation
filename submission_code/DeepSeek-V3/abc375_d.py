# YOUR CODE HERE
from collections import defaultdict

def count_palindromic_triples(S):
    n = len(S)
    # Precompute the frequency of each character at each position
    # We need to find all i < j < k such that S[i] == S[k]
    # So for each j, we need to count the number of pairs (i, k) where i < j < k and S[i] == S[k]
    
    # To do this efficiently, we can precompute the prefix and suffix counts for each character
    prefix_counts = [defaultdict(int) for _ in range(n)]
    suffix_counts = [defaultdict(int) for _ in range(n)]
    
    # Fill prefix_counts
    for i in range(n):
        if i > 0:
            prefix_counts[i] = prefix_counts[i-1].copy()
        prefix_counts[i][S[i]] += 1
    
    # Fill suffix_counts
    for i in range(n-1, -1, -1):
        if i < n-1:
            suffix_counts[i] = suffix_counts[i+1].copy()
        suffix_counts[i][S[i]] += 1
    
    total = 0
    for j in range(1, n-1):
        # For each j, find the number of i < j and k > j such that S[i] == S[k]
        # The total is the sum over all characters c of (number of i < j with S[i] == c) * (number of k > j with S[k] == c)
        for c in prefix_counts[j-1]:
            if c in suffix_counts[j+1]:
                total += prefix_counts[j-1][c] * suffix_counts[j+1][c]
    
    return total

# Read input
S = input().strip()
# Compute and print the result
print(count_palindromic_triples(S))