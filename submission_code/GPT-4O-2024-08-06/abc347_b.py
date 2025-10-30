# YOUR CODE HERE
def count_unique_substrings(S):
    unique_substrings = set()
    n = len(S)
    
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = S[i:j]
            unique_substrings.add(substring)
    
    return len(unique_substrings)

# Read input from standard input
import sys
input = sys.stdin.read
S = input().strip()

# Calculate and print the number of unique substrings
print(count_unique_substrings(S))