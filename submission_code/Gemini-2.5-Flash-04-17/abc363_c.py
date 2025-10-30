# YOUR CODE HERE
import itertools

def is_palindrome(s):
    """Checks if a string is a palindrome."""
    return s == s[::-1]

def contains_k_palindrome(s, k):
    """Checks if a string contains a palindrome substring of length k."""
    n = len(s)
    # If k > n, no substring of length k exists, so it cannot contain one.
    if k > n:
        return False
    # Iterate through all possible start indices for a substring of length k
    # Start index i goes from 0 up to n - k
    for i in range(n - k + 1):
        substring = s[i : i + k]
        if is_palindrome(substring):
            return True
    # No palindrome substring of length k found
    return False

# Read input from stdin
N, K = map(int, input().split())
S = input()

# Get characters as a list to permute.
chars = list(S)

# Use a set to store unique permutations as strings.
distinct_permutations = set()
# itertools.permutations generates permutations of the elements in chars.
# If chars has duplicates, it treats them as distinct objects initially,
# but joining into a string and adding to a set correctly handles distinct string permutations.
for perm_tuple in itertools.permutations(chars):
    distinct_permutations.add("".join(perm_tuple))

# Count permutations that do not contain a palindrome of length K
count_valid = 0
for perm_str in distinct_permutations:
    if not contains_k_palindrome(perm_str, K):
        count_valid += 1

# Print the result to stdout
print(count_valid)