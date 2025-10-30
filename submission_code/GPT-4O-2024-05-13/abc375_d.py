# YOUR CODE HERE
def count_palindromic_triples(S):
    n = len(S)
    count = 0
    
    # Iterate over all possible (i, j, k) triples
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if S[i] == S[k]:
                    count += 1
    
    return count

# Read input
import sys
input = sys.stdin.read().strip()

# Calculate and print the result
result = count_palindromic_triples(input)
print(result)