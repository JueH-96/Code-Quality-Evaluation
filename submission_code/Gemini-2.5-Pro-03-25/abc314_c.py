import sys

# Read input from stdin
# N: length of the string S
# M: number of colors
n, m = map(int, sys.stdin.readline().split())

# S: the string
s = sys.stdin.readline().strip()

# C: list of colors for each character in S (1-based)
c = list(map(int, sys.stdin.readline().split()))

# Create a data structure to store the original indices for each color.
# We use a list of lists, where color_indices[i] holds the 0-based indices
# of characters that have color i+1.
# Initialize m empty lists, one for each color from 1 to M.
color_indices = [[] for _ in range(m)]

# Populate the color_indices list.
# Iterate through the string and its colors.
for idx in range(n):
    # Get the 1-based color value for the character at index idx.
    color_val_1_based = c[idx]
    # Convert the 1-based color value to a 0-based index for our list.
    color_idx_0_based = color_val_1_based - 1
    # Append the original 0-based index (idx) to the list corresponding to its color.
    color_indices[color_idx_0_based].append(idx)

# Create a list to store the characters of the final resulting string.
# Initialize it with N empty strings or placeholders. Using a list allows
# efficient modification at specific indices.
final_chars = [''] * n

# Process each color from 1 to M.
# The loop iterates through the 0-based color indices from 0 to m-1.
for color_idx_0_based in range(m):
    # Get the list of original 0-based indices for the current color.
    # These indices are already sorted because we processed them from left to right (0 to N-1).
    indices = color_indices[color_idx_0_based]

    # Get the number of characters (k) that have this color.
    k = len(indices)

    # Perform the right circular shift only if there are characters of this color.
    # The problem statement guarantees k >= 1 for all colors.
    # If k=1, the character effectively shifts to its own position, resulting in no change.
    if k > 0:
        # --- Perform the right circular shift ---
        # The character originally at the last position (indices[k-1])
        # moves to the first position (indices[0]).

        # Find the original index of the last character in the sequence for this color.
        last_original_index = indices[k - 1]
        # Get the character from the *original* string S at that index.
        char_from_last = s[last_original_index]
        # Find the target index where this character should be placed in the final string.
        first_target_index = indices[0]
        # Place the character in the result list.
        final_chars[first_target_index] = char_from_last

        # The characters originally at positions indices[0] through indices[k-2]
        # shift one position to the right (to indices[1] through indices[k-1], respectively).

        # Iterate through the remaining positions from the second (j=1) to the last (j=k-1).
        for j in range(1, k):
            # Find the original index of the character that moves to the current target position indices[j].
            # This is the character originally at indices[j-1].
            source_original_index = indices[j - 1]
            # Get the character from the *original* string S at that source index.
            char_from_source = s[source_original_index]
            # Determine the target index (indices[j]) where this character should go.
            current_target_index = indices[j]
            # Place the character in the result list.
            final_chars[current_target_index] = char_from_source

# After processing all colors and performing all shifts,
# the final_chars list contains the characters of the resulting string in the correct order.
# Join the characters in the list to form the final string.
result_string = "".join(final_chars)

# Print the final string to standard output.
print(result_string)