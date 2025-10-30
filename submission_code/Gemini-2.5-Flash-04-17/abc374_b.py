import sys

# Read the two strings from standard input
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Get the lengths of the strings
len_S = len(S)
len_T = len(T)

# Determine the minimum length to iterate up to
min_len = min(len_S, len_T)

# Iterate through the strings up to the minimum length
found_difference = False
for i in range(min_len):
    # Compare characters at the current index (0-based)
    if S[i] != T[i]:
        # Found the first difference at index i (0-based)
        # The problem asks for 1-based position, so print i + 1
        print(i + 1)
        found_difference = True
        break # Exit the loop once the first difference is found

# If the loop finished without finding a difference in the common part
if not found_difference:
    # This means the strings match character by character up to min_len
    # Now, check if the lengths are different.
    if len_S != len_T:
        # If lengths are different, the first "difference" is that one string
        # continues while the other ends. This occurs at the position
        # immediately after the end of the shorter string.
        # This position is min_len (which is len_S or len_T, depending on which is shorter)
        # This corresponds to the (min_len + 1)-th character position (1-based).
        print(min_len + 1)
    else:
        # If lengths are the same and no difference was found in the loop,
        # the strings are identical.
        print(0)