import sys
from itertools import permutations

# This program calculates the number of unique permutations of a string S
# that do not contain a palindrome of a specified length K as a substring.

# Read the length of the string N, the length of the palindrome K, and the string S.
try:
    line = sys.stdin.readline()
    if not line:
        # Handle case of empty input
        exit()
        
    N, K = map(int, line.split())
    S = sys.stdin.readline().strip()
except (IOError, ValueError):
    # Handle potential I/O or parsing errors.
    exit()

# Initialize a counter for the permutations that meet the condition.
valid_permutation_count = 0

# `itertools.permutations(S)` generates all permutations of the characters in S.
# If S has duplicate characters, `permutations` will produce arrangements that
# result in the same string. For example, for S="aab", permutations of the
# individual 'a's will yield "aab" multiple times.
# By converting the iterator to a `set`, we ensure that we only process
# each unique permutation string exactly once.
unique_permutations = set(permutations(S))

# Iterate through each unique permutation.
for p_tuple in unique_permutations:
    # Join the tuple of characters to form the permutation string.
    p_str = "".join(p_tuple)
    
    # A flag to determine if the current permutation is valid (i.e., has no
    # palindromic substring of length K).
    is_valid = True
    
    # Check every substring of length K. The number of such substrings is N - K + 1.
    # The loop runs from i = 0 to N - K.
    for i in range(N - K + 1):
        substring = p_str[i : i + K]
        
        # A string is a palindrome if it reads the same forwards and backwards.
        # In Python, this can be easily checked by comparing the string with its reverse.
        if substring == substring[::-1]:
            # If a palindromic substring is found, the permutation is not valid.
            is_valid = False
            # We can break out of the inner loop as we only need to find one such substring.
            break
    
    # If after checking all substrings, the permutation is still considered valid,
    # we increment our counter.
    if is_valid:
        valid_permutation_count += 1

# Print the final count to standard output.
print(valid_permutation_count)