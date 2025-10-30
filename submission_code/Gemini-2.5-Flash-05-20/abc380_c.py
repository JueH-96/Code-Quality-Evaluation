import sys

def solve():
    # Read N and K from the first line
    N, K = map(int, sys.stdin.readline().split())
    # Read the string S from the second line
    S = sys.stdin.readline().strip()

    # Step 1: Find all 1-blocks
    # Store them as (start_index, end_index) tuples using 0-based indexing
    blocks = []
    i = 0
    while i < N:
        if S[i] == '0':
            i += 1
            continue
        
        # Found a '1', which is the start of a potential 1-block
        block_start = i
        # Continue as long as we are within bounds and current char is '1'
        while i < N and S[i] == '1':
            i += 1
        # The block ends at the character just before 'i'
        block_end = i - 1 
        blocks.append((block_start, block_end))

    # Step 2: Identify the (K-1)-th and K-th 1-blocks
    # K is 1-based, so for 0-based list indexing:
    # (K-1)-th block is at index K-2
    # K-th block is at index K-1
    
    prev_block_idx = K - 2
    curr_block_idx = K - 1

    # Extract start and end indices for the two relevant blocks
    l_prev, r_prev = blocks[prev_block_idx] # (K-1)-th block's indices
    l_curr, r_curr = blocks[curr_block_idx] # K-th block's indices

    # Calculate the length of the K-th block
    len_k_block = r_curr - l_curr + 1

    # Step 3: Construct the resulting string T
    # Convert S to a mutable list of characters for easier modification
    res = list(S)

    # Apply Rule 2: Place the K-th block (as '1's) immediately after the (K-1)-th block
    # This range is from (r_prev + 1) up to (r_prev + len_k_block) inclusive.
    # In Python's range(), the end index is exclusive, so we add 1.
    for i in range(r_prev + 1, r_prev + len_k_block + 1):
        res[i] = '1'

    # Apply Rule 3: Fill the 'void' left by the K-th block (and any intermediate characters) with '0's
    # This range is from (r_prev + len_k_block + 1) up to (r_curr) inclusive.
    # In Python's range(), the end index is exclusive, so we add 1.
    for i in range(r_prev + len_k_block + 1, r_curr + 1):
        res[i] = '0'

    # Step 4: Join the list of characters back into a string and print
    sys.stdout.write("".join(res) + "
")

# Call the solve function to execute the program
solve()