# YOUR CODE HERE
import sys

def main():
    """
    Solves the 1122 Sequence problem by identifying blocks of pairs and finding
    the longest sub-segment with unique pair values.
    """
    # Reading input
    try:
        N_str = sys.stdin.readline()
        # Handle cases with empty input files or lines
        if not N_str.strip():
            print(0)
            return
        N = int(N_str)
        
        if N == 0:
            print(0)
            return
            
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handles cases with malformed input.
        print(0)
        return

    # A 1122 sequence must have length at least 2.
    if N < 2:
        print(0)
        return

    max_len = 0
    i = 0
    while i < N - 1:
        # A 1122 sequence must be composed of pairs. We look for blocks of pairs.
        # If A[i] and A[i+1] don't form a pair, a block can't start at i.
        if A[i] != A[i + 1]:
            i += 1
            continue

        # Found the start of a block of pairs at index i.
        start_block_idx = i
        
        # Scan forward to find the end of this contiguous block of pairs.
        j = i + 2
        while j < N - 1 and A[j] == A[j + 1]:
            j += 2
        end_block_idx = j
        
        # The block of pairs is the subarray A[start_block_idx : end_block_idx].
        # A 1122 sequence requires the values forming the pairs to be unique.
        # We extract these values into a new list `pair_values`.
        # For a block (x1,x1, x2,x2, ...), this list will be (x1, x2, ...).
        pair_values = [A[k] for k in range(start_block_idx, end_block_idx, 2)]
        
        # Find the length of the longest contiguous subarray of `pair_values`
        # with unique elements using a sliding window.
        l = 0  # Left pointer of the sliding window
        seen = {}  # A map to store the last seen index: {value: index}
        current_max_unique_len = 0
        for r in range(len(pair_values)):
            val = pair_values[r]
            if val in seen and seen[val] >= l:
                # If `val` is seen within the current window [l, r),
                # shrink the window from the left by moving `l` past the
                # previous occurrence of `val`.
                l = seen[val] + 1
            
            # Update the last seen index for the current value.
            seen[val] = r
            
            # The length of the current unique window is r - l + 1.
            current_max_unique_len = max(current_max_unique_len, r - l + 1)
        
        # The length in the original array is twice the length in `pair_values`.
        # Update the overall maximum length found.
        max_len = max(max_len, 2 * current_max_unique_len)
        
        # Move the main pointer `i` to the end of the processed block.
        i = end_block_idx

    print(max_len)

if __name__ == "__main__":
    main()