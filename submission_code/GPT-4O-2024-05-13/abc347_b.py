# YOUR CODE HERE
def count_unique_substrings(S):
    substrings = set()
    n = len(S)
    
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(S[i:j])
    
    return len(substrings)

# Read input
import sys
input = sys.stdin.read().strip()

# Calculate and print the result
print(count_unique_substrings(input))