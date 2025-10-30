import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()
    # C_values contains 1-indexed colors for characters S_str[0]...S_str[N-1]
    C_values = list(map(int, sys.stdin.readline().split()))

    # Convert string to list of characters for in-place modification
    s_char_list = list(S_str)

    # color_specific_indices[i] will store a list of 0-based original indices 
    # of characters in S_str that are painted with color (i+1).
    # e.g., color_specific_indices[0] is for color 1.
    color_specific_indices = [[] for _ in range(M)] 
    for char_original_idx in range(N):
        # C_values[char_original_idx] is the 1-indexed color of S_str[char_original_idx]
        color_val_1_indexed = C_values[char_original_idx]
        # Store the 0-based original index in the list for the corresponding 0-indexed color
        color_specific_indices[color_val_1_indexed - 1].append(char_original_idx)

    # Process operations for each color, from color 1 up to color M.
    # The loop variable `color_idx_0_based` goes from 0 to M-1,
    # corresponding to processing color (color_idx_0_based + 1).
    for color_idx_0_based in range(M):
        # Get the list of 0-based original indices for the current color.
        # These indices are already sorted as they were added in increasing order.
        indices_this_color = color_specific_indices[color_idx_0_based]
        
        num_chars_this_color = len(indices_this_color)
        
        # If 0 or 1 character has this color, a circular shift doesn't change anything.
        # The problem guarantees that at least one character of S is painted in each color,
        # so num_chars_this_color will be at least 1.
        # Thus, we only need to act if num_chars_this_color > 1.
        if num_chars_this_color > 1:
            # Take a snapshot of the characters at these positions *before* modification for this color.
            # These are the "old" values that will be shifted.
            snapshot_of_chars_to_shift = [s_char_list[idx] for idx in indices_this_color]
            
            # Perform the right circular shift.
            # The character S[p_k] (old) moves to position p_1.
            # The character S[p_1] (old) moves to position p_2.
            # ...
            # The character S[p_{k-1}] (old) moves to position p_k.
            #
            # In terms of 0-indexed lists `indices_this_color` (length k) and `snapshot_of_chars_to_shift` (length k):
            #   s_char_list[indices_this_color[j]] (new value)
            #   will be taken from
            #   snapshot_of_chars_to_shift[(j - 1 + k) % k] (old value)
            # where k = num_chars_this_color.
            
            for j in range(num_chars_this_color):
                # Determine which character from the snapshot moves to s_char_list[indices_this_color[j]].
                # This is the (j-1)-th character in the snapshot (0-indexed), wrapping around for j=0.
                # e.g., for j=0, (0 - 1 + k) % k = (k-1)%k = k-1 (last element of snapshot).
                # e.g., for j=1, (1 - 1 + k) % k = 0%k = 0 (first element of snapshot).
                idx_in_snapshot = (j - 1 + num_chars_this_color) % num_chars_this_color
                char_to_place = snapshot_of_chars_to_shift[idx_in_snapshot]
                
                # This is the actual index in s_char_list to update.
                target_idx_in_s_char_list = indices_this_color[j]
                s_char_list[target_idx_in_s_char_list] = char_to_place

    # After all operations, join the list of characters back into a string and print.
    print("".join(s_char_list))

if __name__ == '__main__':
    main()