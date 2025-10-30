# YOUR CODE HERE
import sys

# Function to check if two characters are similar according to the rules:
# - x == y
# - one is '1', other is 'l'
# - one is '0', other is 'o'
# This can be simplified by normalizing 'l' to '1' and 'o' to '0'
# and then checking if the normalized characters are equal.
def normalize(char):
    """ Normalizes 'l' to '1' and 'o' to '0'. Other characters remain unchanged. """
    if char == 'l':
        return '1'
    if char == 'o':
        return '0'
    return char

# Read input from stdin
# Read the length of the strings
N = int(sys.stdin.readline())
# Read string S
S = sys.stdin.readline().strip()
# Read string T
T = sys.stdin.readline().strip()

# Assume the strings are similar initially
is_similar_strings = True

# Iterate through the strings character by character from index 0 to N-1
for i in range(N):
    # Get the characters at the current index i
    s_char = S[i]
    t_char = T[i]

    # Normalize both characters
    normalized_s_char = normalize(s_char)
    normalized_t_char = normalize(t_char)

    # Check if the normalized characters are different.
    # If they are different, it means the original characters are not similar
    # according to the problem definition.
    if normalized_s_char != normalized_t_char:
        # If any pair of characters is not similar, the strings are not similar
        is_similar_strings = False
        # No need to check further, we can break the loop
        break

# Print the final result to stdout
if is_similar_strings:
    # If the loop completed without finding any non-similar pair, the strings are similar
    print("Yes")
else:
    # If a non-similar pair was found, the strings are not similar
    print("No")