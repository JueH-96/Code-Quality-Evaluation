# YOUR CODE HERE
import sys
from collections import defaultdict

def count_palindromic_triples(S):
    n = len(S)
    count = 0
    char_indices = defaultdict(list)

    # Collect indices of each character
    for i in range(n):
        char_indices[S[i]].append(i)

    # Check for palindromic triples
    for char, indices in char_indices.items():
        m = len(indices)
        if m >= 3:
            # Count triples (i, j, k) where i < j < k and S[i] == S[j] == S[k]
            count += (m * (m - 1) * (m - 2)) // 6

    return count

# Read input
S = sys.stdin.read().strip()

# Calculate and print the result
result = count_palindromic_triples(S)
print(result)