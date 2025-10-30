# YOUR CODE HERE
import sys

# Read the input string from stdin
S = sys.stdin.readline().strip()

# Initialize a set to store distinct substrings
distinct_substrings = set()

# Get the length of the string
N = len(S)

# Iterate through all possible start indices
for i in range(N):
    # Iterate through all possible end indices (inclusive)
    # The end index j ranges from i to N-1.
    for j in range(i, N):
        # Extract the substring from index i to j (inclusive).
        # In Python slicing S[start:end], 'end' is the first index NOT included.
        # So, S[i : j + 1] includes characters from index i up to index j.
        substring = S[i : j + 1]
        # Add the substring to the set. Sets automatically handle duplicates.
        distinct_substrings.add(substring)

# The number of different non-empty substrings is the size of the set.
result = len(distinct_substrings)

# Print the result to stdout.
print(result)