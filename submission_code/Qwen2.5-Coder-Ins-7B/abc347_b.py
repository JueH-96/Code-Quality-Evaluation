# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Initialize a set to store unique substrings
unique_substrings = set()

# Iterate over all possible starting points of substrings
for i in range(len(S)):
    # Iterate over all possible ending points of substrings
    for j in range(i + 1, len(S) + 1):
        # Add the substring to the set of unique substrings
        unique_substrings.add(S[i:j])

# Print the number of unique substrings
print(len(unique_substrings))