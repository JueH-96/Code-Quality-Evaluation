# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, performs the specified string transformation, and prints the result.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem inputs
    try:
        N, K = map(int, input().split())
        S = input().strip()
    except (IOError, ValueError):
        # Graceful exit for empty input
        return

    # --- Step 1: Find all 1-blocks ---
    # Store the 0-based (start, end) indices of each block.
    blocks = []
    in_block = False
    start_idx = -1
    for i, char in enumerate(S):
        if char == '1' and not in_block:
            # Found the start of a new block
            in_block = True
            start_idx = i
        elif char == '0' and in_block:
            # Found the end of the current block
            in_block = False
            blocks.append((start_idx, i - 1))
    
    # If a block extends to the end of the string, it needs to be added.
    if in_block:
        blocks.append((start_idx, N - 1))

    # --- Step 2: Locate key block indices ---
    # The problem uses 1-based K, so we access our 0-indexed list at K-2 and K-1.
    # It's guaranteed that at least K blocks exist.
    r_km1 = blocks[K - 2][1]  # 0-based end index of the (K-1)-th block
    r_k = blocks[K - 1][1]    # 0-based end index of the K-th block

    # --- Step 3: Rearrange the middle segment ---
    # The operation is equivalent to rearranging the substring S[r_km1 + 1 : r_k + 1]
    # by moving all its '1's to the front.
    s_list = list(S)
    
    # Define the 0-based indices of the segment to be rearranged.
    start_rearrange = r_km1 + 1
    end_rearrange = r_k
    
    # Count the number of '1's in this segment.
    num_ones = 0
    for i in range(start_rearrange, end_rearrange + 1):
        if s_list[i] == '1':
            num_ones += 1
            
    # Overwrite the segment in-place: '1's first, then '0's.
    for i in range(start_rearrange, start_rearrange + num_ones):
        s_list[i] = '1'
    for i in range(start_rearrange + num_ones, end_rearrange + 1):
        s_list[i] = '0'
        
    # --- Step 4: Print the final string ---
    print("".join(s_list))

# Run the solution
solve()