import sys

def compute_pi(T):
    """
    Computes the KMP border array (also known as pi or LPS array) for string T.
    pi[i] is the length of the longest proper prefix of T[0...i] that is also a suffix of T[0...i].
    """
    m = len(T)
    pi = [0] * m
    j = 0  # length of the previous longest prefix suffix
    for i in range(1, m):
        # If mismatch, reduce j by looking up the pi value for the previous prefix
        while j > 0 and T[i] != T[j]:
            j = pi[j - 1]
        # If match, increase j
        if T[i] == T[j]:
            j += 1
        pi[i] = j
    return pi

# Read input from stdin
S = sys.stdin.readline().strip()
n = len(S)

# Construct the string T = S_reversed + '#' + S
# The KMP border value at the end of T gives the length of the longest palindromic suffix of S.
S_rev = S[::-1]
T = S_rev + '#' + S

# Compute KMP pi array for T
pi = compute_pi(T)

# The length of the longest palindromic suffix of S is the last value in the pi array
# pi[len(T)-1] = pi[2n]
p = pi[len(T) - 1]

# The longest palindromic suffix of S has length p, starting at index n-p.
# The prefix of S that is NOT part of this longest palindromic suffix is S[0..n-p-1].
# This prefix needs to be reversed and appended to S to form the shortest palindrome.
# The prefix is S[:n-p] in Python slice notation.
prefix_to_reverse = S[:n - p]

# Reverse the prefix
# Use slice [::-1] for efficient reversal
reversed_prefix = prefix_to_reverse[::-1]

# Construct the shortest palindrome
# Palindrome = S + (reversed prefix)
result = S + reversed_prefix

# Print the result to stdout
sys.stdout.write(result + '
')