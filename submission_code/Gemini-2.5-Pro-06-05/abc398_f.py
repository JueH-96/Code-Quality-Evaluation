import sys

# Read the input string from standard input.
# Using sys.stdin.readline() for better performance with large inputs.
S = sys.stdin.readline().strip()
n = len(S)

# The reverse of the input string.
s_rev = S[::-1]

# To find the longest palindromic suffix of S, we use a KMP-based technique.
# We construct a temporary string by concatenating the reversed string, a unique
# separator, and the original string.
# The separator must be a character not present in the original string's alphabet.
# Since S contains only uppercase English letters, '#' is a safe choice.
temp_str = s_rev + '#' + S
m = len(temp_str)

# Compute the Longest Proper Prefix which is also a Suffix (LPS) array
# for the temporary string. This is the core of the KMP preprocessing step.
lps = [0] * m
length = 0  # Length of the previous longest prefix suffix
i = 1
while i < m:
    if temp_str[i] == temp_str[length]:
        length += 1
        lps[i] = length
        i += 1
    else:
        if length != 0:
            # Fall back to the LPS value of the previous character in the prefix.
            length = lps[length - 1]
        else:
            # No common prefix/suffix, lps[i] is 0.
            # lps[i] is already 0 from initialization.
            i += 1

# The last value in the LPS array gives the length of the longest suffix of S
# that is also a prefix of S_rev, which is the length of the longest
# palindromic suffix of S.
len_pal_suffix = lps[-1]

# The part to append to S is the reverse of the non-palindromic prefix of S.
# The non-palindromic prefix has length n - len_pal_suffix.
to_append = S[:n - len_pal_suffix][::-1]

# The final result is S followed by the reversed prefix.
print(S + to_append)