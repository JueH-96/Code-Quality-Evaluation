import sys

# The KMP prefix function (failure function) implementation
# pi[i] is the length of the longest proper prefix of text[0...i]
# that is also a suffix of text[0...i].
def compute_kmp_pi(text):
    n = len(text)
    pi = [0] * n
    # Iterate from the second character
    for i in range(1, n):
        # Start with the previously computed border length
        j = pi[i - 1]
        # While the current character does not match the character after the border
        # and the border length is not zero, backtrack using the pi table.
        # text[j] is the character immediately following the border prefix text[0...j-1]
        # we are trying to match text[i] with.
        while j > 0 and text[i] != text[j]:
            j = pi[j - 1]
        # If the current character matches the character at the border index j
        # (which is the character that would extend the border)
        if text[i] == text[j]:
            # Extend the border length
            j += 1
        # Store the new border length for text[0...i]
        pi[i] = j
    return pi

# Read input string from stdin
S = sys.stdin.readline().strip()

n = len(S)

# To find the shortest palindrome with S as a prefix, we need to append
# the minimum number of characters to S.
# This minimum number of characters is achieved by finding the longest suffix
# of S that is already "palindromic" with respect to the prefix that needs
# to be appended.
# More formally, we find the largest k such that the suffix of S of length k
# is equal to the prefix of S[::-1] of length k.
# The characters S[0...n-k-1] (the prefix of S *not* covered by the suffix)
# need to be reversed and appended to S.
# We use the KMP algorithm on the string S[::-1] + '#' + S to find this k.
# The pi value at the last index of this combined string gives the length
# of the longest prefix of S[::-1] that is also a suffix of S.

S_R = S[::-1]

# Use a character not in S as a separator. Uppercase English letters are A-Z.
# '#' is a safe choice.
separator = '#'
combined_string = S_R + separator + S

# Compute the KMP prefix function for the combined string.
# The pi value at the last index tells us the length of the longest
# proper prefix of combined_string that is also a suffix of combined_string.
# T = S_R + '#' + S
# prefix = T[0...k-1] = S_R[0...k-1]
# suffix = T[len(T)-k...len(T)-1] = S[n-k...n-1]
# The longest match length k means S_R[0...k-1] == S[n-k...n-1].
# This is the length of the longest suffix of S that is a prefix of S_R.
pi = compute_kmp_pi(combined_string)

# k is the length of the suffix of S that is a prefix of S_R.
# This means S[n-k:] is the part of S that is already aligned
# with the beginning of S_R (its reverse).
# The part of S that is NOT aligned is S[0:n-k].
# These characters need to be reversed and appended to S.
k = pi[len(combined_string) - 1]

# Extract the prefix of S that needs to be reversed
prefix_part_to_reverse = S[0 : n - k]

# Reverse the prefix part
suffix_to_add = prefix_part_to_reverse[::-1]

# The shortest palindrome is S followed by the reversed prefix part
result = S + suffix_to_add

# Print the result to stdout
print(result)