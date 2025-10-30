import sys

def change_case(char):
    """Changes the case of an English letter."""
    if 'a' <= char <= 'z':
        return char.upper()
    elif 'A' <= char <= 'Z':
        return char.lower()
    return char # Return as is for non-letters, though problem constraints say only letters

def solve():
    """Solves the problem as per the specification."""
    # Read initial string S and number of queries Q
    s_initial = sys.stdin.readline().strip()
    q = int(sys.stdin.readline())

    # Read query indices K_i
    k_queries = list(map(int, sys.stdin.readline().split()))

    l0 = len(s_initial)
    results = []

    # Process each query
    for k_query in k_queries:
        k = k_query - 1 # Convert 1-based index to 0-based index
        p = 0 # Initialize total flips count

        # Find the starting level (smallest i) such that S_i is guaranteed
        # to contain the index k. |S_i| = 2^i * L0. We need 2^i * L0 > k.
        current_level = 0
        current_len = l0
        
        # Check if the index k is within the initial string S_0
        # This condition is equivalent to k < L0.
        # If k < L0, the index is in S_0 from the start.
        # If k >= L0, we need to find the smallest i > 0 such that 2^i * L0 > k.
        # The loop below finds this smallest i.
        while current_len <= k:
            # Check for potential overflow before doubling current_len
            # Max k is 10^18 - 1. Max L0 is 2e5.
            # current_len can grow up to approximately 2 * (10^18 - 1).
            # Python's arbitrary precision integers handle this.
            current_len *= 2
            current_level += 1
        
        # Now, current_level is the smallest integer i such that 2^i * L0 > k.
        # We trace back from this level down to level 0 (S_0).
        # At each level i (from current_level down to 1), we check if the index k
        # falls into the second half (the T_{i-1} part) of S_i = S_{i-1} T_{i-1}.
        # The length of S_{i-1} is 2^(i-1) * L0.
        # If k >= |S_{i-1}|, the character at index k in S_i is the case-changed
        # version of the character at index (k - |S_{i-1}|) in S_{i-1}.
        # So, we update k and increment the flip count p.
        # If k < |S_{i-1}|, the character at index k in S_i is the same as the
        # character at index k in S_{i-1}. No update to k or p needed, just move to S_{i-1}.
        
        while current_level > 0:
            # Calculate length of S_{current_level - 1}
            # Use bit shift for 2^(current_level - 1)
            # current_level - 1 can be up to ~60. 1 << 60 is large.
            # The product (1 << (current_level - 1)) * l0 can exceed 64-bit int if l0 is large.
            # Python handles this with arbitrary precision integers.
            len_s_prev = (1 << (current_level - 1)) * l0

            if k >= len_s_prev:
                k -= len_s_prev
                p += 1
            
            current_level -= 1 # Move down to the next level (S_{current_level - 1})

        # After the loop, current_level is 0, meaning k is now the 0-based index
        # in the original string S_0. The total number of flips accumulated is p.
        
        # Get the character from S_0 at the final index k
        final_char = s_initial[k]

        # Apply case changes if p is odd
        if p % 2 == 1:
            final_char = change_case(final_char)

        results.append(final_char)

    # Print the results separated by spaces
    print(*results)

# Run the solver function
solve()