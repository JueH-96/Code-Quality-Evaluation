import sys
from collections import defaultdict

def count_palindromic_triples(S):
    n = len(S)
    count = 0

    # Dictionary to store the last two occurrences of each character
    last_two_occurrences = defaultdict(list)

    for i in range(n):
        char = S[i]
        if len(last_two_occurrences[char]) == 2:
            last_two_occurrences[char].pop(0)
        last_two_occurrences[char].append(i)

        # Check for palindromic triples
        for j in last_two_occurrences[char]:
            if j > 0 and j < i:
                count += j

    return count

# Read input from stdin
S = sys.stdin.read().strip()

# Calculate and print the result
result = count_palindromic_triples(S)
print(result)