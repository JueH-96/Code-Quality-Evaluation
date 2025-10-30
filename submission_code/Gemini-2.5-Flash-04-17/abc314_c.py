# YOUR CODE HERE
import sys
from collections import defaultdict

# Read input from stdin
input = sys.stdin.readline

# Read N (length of string) and M (number of colors)
N, M = map(int, input().split())

# Read the string S and convert it to a list of characters for mutability.
# .strip() is used to remove potential trailing newline characters from the input line.
S = list(input().strip())

# Read the list of colors C. C[i] is the color of the character S[i].
# The input line contains N space-separated integers.
C = list(map(int, input().split()))

# Group indices by color.
# color_indices is a dictionary where keys are colors (from 1 to M)
# and values are lists of 0-based indices in S where characters of that color appear.
# The indices stored in the list for each color are automatically in increasing order
# because we iterate through the string's indices from 0 to N-1.
color_indices = defaultdict(list)
for i in range(N):
    # C[i] holds the color for the character at index i in the string S.
    color_indices[C[i]].append(i)

# Perform the specified operation for each color from 1 to M, in increasing order.
# The problem guarantees that each color from 1 to M appears in C at least once.
# This implies that for each color_i in the loop range, color_indices[color_i]
# will exist and its list of indices will be non-empty (length k >= 1).
for color_i in range(1, M + 1):
    # Get the list of 0-based indices for the current color.
    indices = color_indices[color_i]
    k = len(indices) # Number of characters of this color. We know k >= 1.

    # Perform a right circular shift by 1 on the characters located at these 'indices'.
    # This shift is applied to the current state of the string S.
    # The character at the last index in the 'indices' list (indices[k-1]) moves to
    # the first index (indices[0]).
    # The character at indices[j-1] moves to indices[j] for j from 1 to k-1.

    # 1. Store the character that is currently at the last relevant index (indices[k-1]).
    # This character will be moved to the first relevant index (indices[0]) after shifting others.
    char_to_move_to_first = S[indices[k-1]]

    # 2. Shift characters from right to left.
    # We iterate through the 'indices' list from the second-to-last element's position (k-2)
    # down to the first element's position (0).
    # The character at indices[j-1] (which is to the "left" in the original string order)
    # moves to the position indices[j].
    # The loop `range(k - 1, 0, -1)` iterates j from k-1 down to 1.
    # For j=k-1, we copy S[indices[k-2]] to S[indices[k-1]].
    # For j=k-2, we copy S[indices[k-3]] to S[indices[k-2]].
    # ...
    # For j=1, we copy S[indices[0]] to S[indices[1]].
    # This correctly shifts the characters at indices[0] through indices[k-2] one position to the right.
    for j in range(k - 1, 0, -1):
        S[indices[j]] = S[indices[j-1]]

    # 3. Place the stored character (originally at indices[k-1]) into the first index (indices[0]).
    S[indices[0]] = char_to_move_to_first

# After processing all colors from 1 to M, the string S is in its final state.
# Convert the modified list of characters back to a string and print it to standard output.
print("".join(S))