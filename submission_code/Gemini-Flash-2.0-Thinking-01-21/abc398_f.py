# YOUR CODE HERE
import sys

def compute_pi(text):
    """
    Computes the KMP prefix function (border array) for a string.
    pi[i] is the length of the longest proper prefix of text[0...i]
    that is also a suffix of text[0...i].
    """
    n = len(text)
    pi = [0] * n
    j = 0  # length of the previous longest prefix suffix

    # pi[0] is always 0, start from i=1
    for i in range(1, n):
        # If there is a mismatch after j characters,
        # the next character to match in the prefix is text[j].
        # We need to backtrack in the prefix using the pi array
        # until we find a character that matches text[i] or j becomes 0.
        while j > 0 and text[i] != text[j]:
            j = pi[j-1]

        # If characters match, increment the length of the current border
        if text[i] == text[j]:
            j += 1

        # Store the length of the border ending at index i
        pi[i] = j

    return pi

# Read the input string S
S = sys.stdin.readline().strip()
n = len(S)

# Construct the string T = reverse(S) + '#' + S
# The KMP prefix function computed on T helps find the length of
# the longest suffix of S that is also a prefix of reverse(S).
# This length is exactly the length of the longest palindromic suffix of S.
# Using a character not in S (like '#') ensures the match does not cross the separator.
T = S[::-1] + '#' + S
N = len(T)

# Compute the KMP prefix function for T
pi = compute_pi(T)

# The length of the longest palindromic suffix of S is pi[N-1].
# Let L = pi[N-1].
# The string T = reverse(S)[0...n-1] + '#' + S[0...n-1].
# The longest proper prefix of T that is also a suffix has length L.
# This prefix is T[0...L-1] = reverse(S)[0...L-1].
# This suffix is T[N-L...N-1] = S[n-L...n-1].
# Since they are equal, reverse(S)[0...L-1] == S[n-L...n-1].
# The substring reverse(S)[0...L-1] is S[n-1]S[n-2]...S[n-L].
# The substring S[n-L...n-1] is S[n-L]S[n-L+1]...S[n-1].
# The equality S[n-1]S[n-2]...S[n-L] == S[n-L]S[n-L+1]...S[n-1] means
# S[n-L]S[n-L+1]...S[n-1] is a palindrome.
# So, L = pi[N-1] is the length of the longest palindromic suffix of S.
k_len = pi[N-1]

# The shortest palindrome with S as a prefix is formed by taking S
# and appending the reverse of the prefix of S that does not overlap
# with the longest palindromic suffix.
# The longest palindromic suffix starts at index n - k_len.
# The part of S before this suffix is S[0 : n - k_len].
# We need to reverse this part and append it to S.
prefix_to_reverse = S[0 : n - k_len]

# The characters to append is the reverse of this prefix
suffix_to_append = prefix_to_reverse[::-1]

# The shortest palindrome is S + suffix_to_append
result = S + suffix_to_append

# Print the result
print(result)