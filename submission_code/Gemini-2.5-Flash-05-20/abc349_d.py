import sys

def solve():
    L, R = map(int, sys.stdin.readline().split())

    result_segments = []
    current_L = L

    # Loop until the current_L reaches R
    while current_L < R:
        # We need to find the largest 'i' such that 2^i is the length of a good segment
        # S(current_L, current_L + 2^i) that fits within [L, R)

        chosen_i = -1 # Will store the exponent 'i' for the selected length 2^i

        # Iterate 'i' from the maximum possible (e.g., 60 for 2^60) down to 0.
        # This ensures we find the largest valid 2^i first.
        for i in range(60, -1, -1):
            length = 1 << i # Calculate 2^i

            # Condition 1: current_L must be a multiple of the segment length (2^i)
            # This is key for S(l, r) = S(2^i * j, 2^i * (j+1))
            # (current_L % length == 0) is true for 0 % X == 0, which correctly handles current_L = 0.
            is_multiple = (current_L % length == 0)

            # Condition 2: The end of the segment must not exceed R
            within_bounds = (current_L + length <= R)

            # If both conditions are met, this 'i' is a valid candidate.
            # Since we iterate 'i' downwards, the first one found is the largest.
            if is_multiple and within_bounds:
                chosen_i = i
                break # Found the largest valid 'i', so we stop searching

        # At this point, chosen_i must have a valid value (at least 0, because length 1 is always possible if current_L < R)
        selected_length = 1 << chosen_i
        next_L = current_L + selected_length

        # Add the found segment to our list
        result_segments.append((current_L, next_L))

        # Move to the end of the newly added segment
        current_L = next_L

    # Print the total number of segments
    sys.stdout.write(str(len(result_segments)) + "
")

    # Print each segment (l, r)
    for l_val, r_val in result_segments:
        sys.stdout.write(f"{l_val} {r_val}
")

# Call the solve function to run the program
solve()