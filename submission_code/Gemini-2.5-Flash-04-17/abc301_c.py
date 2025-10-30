import sys
from collections import Counter

# Read inputs
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Define atcoder characters that can replace '@'
atcoder_chars = set('atcoder')

# Count character frequencies for both strings
# Counter handles characters not present by returning 0 when get() is used
count_S = Counter(S)
count_T = Counter(T)

# Track the total number of specific 'atcoder' characters that S needs
# to generate from its '@'s to match T's excess.
# If count_T[c] > count_S[c] for an atcoder char 'c', S needs count_T[c] - count_S[c] more 'c's.
# These must come from S's '@'s.
s_at_must_convert_sum = 0

# Check character counts for all lowercase letters 'a' through 'z'
possible = True
for char_code in range(ord('a'), ord('z') + 1):
    char = chr(char_code)
    
    count_s_char = count_S.get(char, 0)
    count_t_char = count_T.get(char, 0)

    if char not in atcoder_chars:
        # Non-atcoder characters must have equal counts in both strings
        if count_s_char != count_t_char:
            possible = False
            break # Mismatch found, no need to check further characters
    else:
        # Atcoder characters can be replaced by '@'.
        # If S has fewer of an atcoder char than T, S must use its '@'s
        # to make up the difference. This is a required conversion for S.
        if count_s_char < count_t_char:
            s_at_must_convert_sum += (count_t_char - count_s_char)
        # If count_s_char > count_t_char, T has fewer of this char than S.
        # T must use its '@'s to make up the difference.
        # We don't need to calculate a separate sum for T's required conversions
        # because the total number of (atcoder characters + '@') is conserved
        # in both strings (since non-atcoder counts match and lengths are equal).
        # If S can cover its minimum required conversions, T can also cover its minimum
        # required conversions, and the remaining '@'s in S and T will match.

# If a mismatch was found for non-atcoder characters, it's impossible to win
if not possible:
    print("No")
else:
    # Non-atcoder counts matched.
    # Now check if S has enough '@'s to cover the minimum required conversions
    # identified in s_at_must_convert_sum.
    # If this condition is met, it guarantees that T also has enough '@'s
    # for its required conversions, and the remaining '@'s can be matched.
    # count_S.get('@', 0) handles the case where S has no '@'s.
    if s_at_must_convert_sum <= count_S.get('@', 0):
        print("Yes")
    else:
        print("No")