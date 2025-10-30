import sys

def query(i, j):
    """
    Sends a query to the judge and reads the response.
    Args:
        i: Non-negative integer for block size 2^i.
        j: Non-negative integer for block index.
    Returns:
        The remainder when A_l + ... + A_r is divided by 100,
        where l = 2^i * j and r = 2^i * (j+1) - 1.
    """
    print(f"? {i} {j}")
    sys.stdout.flush() # Ensure the query is sent immediately
    
    T = int(sys.stdin.readline())
    
    if T == -1:
        # If the judge returns -1, the program is considered incorrect.
        # Terminate immediately.
        sys.exit(0) 
        
    return T

def solve():
    N, L, R = map(int, sys.stdin.readline().split())

    total_sum_mod_100 = 0
    
    # current_L_block_idx and current_R_block_idx represent the range of block indices
    # at the current 'level' of the segment tree.
    # Initially, level 0 corresponds to blocks of size 2^0 = 1,
    # so current_L_block_idx and current_R_block_idx are simply the actual indices L and R.
    current_L_block_idx = L
    current_R_block_idx = R
    
    # current_level corresponds to 'i' in the query ? i j, representing a block size of 2^i.
    current_level = 0 

    # This loop processes the range [L, R] by iteratively "moving up" the segment tree.
    # It identifies blocks of size 2^current_level from the left and right ends of the current range,
    # sums them, and then adjusts the range and increments the level.
    while current_L_block_idx <= current_R_block_idx:
        # Process the left boundary (current_L_block_idx):
        # If current_L_block_idx is odd (at the current_level), it means this block cannot be combined
        # with the next block to form a larger block at current_level+1 that starts at an even multiple
        # of (2^(current_level+1)).
        # So, we must query the block of size 2^current_level starting at current_L_block_idx.
        if current_L_block_idx % 2 != 0:
            # The 'j' parameter for query(i, j) is the starting block index at level 'i'.
            # Here, 'i' is current_level, and 'j' is current_L_block_idx.
            q_val = query(current_level, current_L_block_idx)
            total_sum_mod_100 = (total_sum_mod_100 + q_val) % 100
            current_L_block_idx += 1 # Move to the next block at this level

        # Process the right boundary (current_R_block_idx):
        # We need to check if current_L_block_idx has crossed current_R_block_idx after processing the left side.
        if current_L_block_idx > current_R_block_idx:
            break

        # If current_R_block_idx is even (at the current_level), it means this block cannot be combined
        # with the previous block to form a larger block that ends on a multiple of (2^(current_level+1)) - 1.
        # So, we must query the block of size 2^current_level ending at current_R_block_idx.
        if current_R_block_idx % 2 == 0:
            # The 'j' parameter for query(i, j) is the starting block index at level 'i'.
            # Here, 'i' is current_level, and 'j' is current_R_block_idx.
            q_val = query(current_level, current_R_block_idx)
            total_sum_mod_100 = (total_sum_mod_100 + q_val) % 100
            current_R_block_idx -= 1 # Move to the previous block at this level
        
        # If current_L_block_idx is now greater than current_R_block_idx, we have covered the entire range.
        if current_L_block_idx > current_R_block_idx:
            break
        
        # At this point, both current_L_block_idx is even and current_R_block_idx is odd.
        # This implies that the remaining sub-range [current_L_block_idx, current_R_block_idx] 
        # (at current_level, meaning each element represents a 2^current_level sized block)
        # can be entirely composed of blocks of size 2 * 2^current_level (i.e., 2^(current_level+1)).
        # We effectively "move up" one level in the segment tree, adjusting our block indices
        # and incrementing the level.
        current_L_block_idx //= 2
        current_R_block_idx //= 2
        current_level += 1

    # Once the loop finishes, total_sum_mod_100 holds the remainder.
    print(f"! {total_sum_mod_100}")
    sys.stdout.flush() # Ensure the final answer is sent

# Call the solver function to start the interaction
solve()