import sys

def compute_lps_array(pattern: str) -> list[int]:
    m = len(pattern)
    # m will be at least 3 in this problem (N>=1, so pattern length 2N+1 >= 3)
    lps = [0] * m  # lps[0] is always 0
    
    length = 0  # length of the previous longest prefix that is also a suffix
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
                # We do not increment i here; try again with shorter 'length'
            else:
                lps[i] = 0 # No common prefix-suffix found of length > 0
                i += 1
    return lps

# Read input string S
S = sys.stdin.readline().strip()
N = len(S)

# Constraints: 1 <= N <= 500,000. N is never 0.

# Reverse S
S_reversed = S[::-1]

# If S is already a palindrome, S_reversed == S.
# In this case, T_KMP = S + "#" + S.
# The lps_values[-1] will be N.
# So, l = N.
# prefix_of_S_to_mirror = S[0 : N - N] = S[0:0] = "" (empty string).
# string_to_append = "" (empty string reversed).
# final_palindrome = S + "" = S.
# This correctly handles cases where S is already a palindrome.

# Construct the temporary string for KMP's LPS calculation
# The separator "#" must not be in S. S contains uppercase English letters.
temp_str = S_reversed + "#" + S 

# Compute LPS array for temp_str
lps_values = compute_lps_array(temp_str)

# The last value in lps_values is the length of the longest palindromic suffix of S
# (which is equivalent to the length of the longest palindromic prefix of S_reversed)
l = lps_values[-1]

# The part of S that is not part of this longest palindromic suffix
# is S[0 : N-l]. This is the part that needs to be reversed and appended.
prefix_of_S_to_mirror = S[0 : N - l]

# Reverse this prefix
string_to_append = prefix_of_S_to_mirror[::-1]

# Construct the final shortest palindrome
final_palindrome = S + string_to_append

# Print the result
sys.stdout.write(final_palindrome + "
")