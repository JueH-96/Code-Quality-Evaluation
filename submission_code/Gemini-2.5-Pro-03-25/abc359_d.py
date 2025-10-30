# YOUR CODE HERE
# Ensure to find the number of good strings modulo 998244353.
import sys

# Use sys.stdin.readline for faster I/O on online judges
# Use input() if running locally and pasting input

def solve():
    # Read input N and K
    N, K = map(int, sys.stdin.readline().split())
    # Read the input string S
    S = sys.stdin.readline().strip()
    # Define the modulus
    MOD = 998244353

    # Precompute palindrome checks for all possible binary strings of length K
    # is_palindrome_K[state_int] will be True if the K-bit integer state_int represents a palindrome.
    # We map 'A' to 0 and 'B' to 1. The integer state_int represents the binary string read naturally,
    # where the least significant bit corresponds to the last character of the string.
    # Example: K=3, string "ABA" -> binary 010 -> integer 2.
    is_palindrome_K = [False] * (1 << K)

    def check_pal(state_int, k):
        """
        Checks if the k-bit integer state_int represents a palindrome string.
        Compares bit p with bit k-1-p for p from 0 to k//2 - 1.
        """
        for p in range(k // 2):
            # Extract bit at position p (0 is the least significant bit)
            bit_p = (state_int >> p) & 1
            # Extract bit at position k-1-p
            bit_k_minus_1_minus_p = (state_int >> (k - 1 - p)) & 1
            # If any pair doesn't match, it's not a palindrome
            if bit_p != bit_k_minus_1_minus_p:
                return False
        # If all pairs match, it is a palindrome
        return True

    # Populate the precomputed table for palindrome checks
    for state_int in range(1 << K):
        if check_pal(state_int, K):
            is_palindrome_K[state_int] = True

    # Create a bitmask to keep integers within K bits. This is (2^K - 1).
    mask = (1 << K) - 1

    # Initialize DP state.
    # Phase 1: Compute all valid prefixes of length K satisfying character constraints from S.
    # Use a dictionary `dp_init` where key is the prefix string and value is the count of ways to form it.
    # Start with an empty prefix having count 1.
    dp_init = {"": 1} 
    
    # Iterate from length 1 to K
    for i in range(1, K + 1):
        # Dictionary to store counts for prefixes of length i
        new_dp_init = {}
        # Get the character constraint from S at the current position i (S is 0-indexed)
        s_char = S[i-1] 
        
        # Iterate through all valid prefixes of length i-1
        for prefix, count in dp_init.items():
            # Optimization: Skip if count is 0 (no ways to reach this prefix)
            if count == 0: continue 
            
            # Try appending 'A' to the prefix
            if s_char == '?' or s_char == 'A':
                new_prefix = prefix + 'A'
                # Update count for the new prefix, summing up contributions modulo MOD
                current_count = new_dp_init.get(new_prefix, 0)
                new_dp_init[new_prefix] = (current_count + count) % MOD
                
            # Try appending 'B' to the prefix
            if s_char == '?' or s_char == 'B':
                new_prefix = prefix + 'B'
                 # Update count for the new prefix, summing up contributions modulo MOD
                current_count = new_dp_init.get(new_prefix, 0)
                new_dp_init[new_prefix] = (current_count + count) % MOD

        # Update the dictionary for the next length
        dp_init = new_dp_init
        # After loop i, dp_init contains counts for all valid prefixes of length i

    # Convert the final prefixes from the initialization phase to the main DP base state `dp_curr`.
    # `dp_curr` is an array (list) representing DP states for index K.
    # The index `state_int` corresponds to the integer representation of the K-length suffix T[1..K].
    # Size of the array is 2^K, initialized to all zeros.
    dp_curr = [0] * (1 << K) 

    # Populate `dp_curr` based on the valid prefixes of length K found in `dp_init`.
    for prefix_K, count in dp_init.items(): # Iterate through computed prefixes of length K
        if count == 0: continue

        # Convert the prefix_K string (e.g., "ABA") to its integer representation `state_int` (e.g., 2).
        state_int = 0
        for char_idx in range(K):
            char = prefix_K[char_idx]
            # Shift state_int left by 1 bit to make space for the next character's bit.
            state_int <<= 1 
            # If the character is 'B' (which maps to 1), set the least significant bit to 1.
            if char == 'B':
                state_int |= 1
        
        # Check if the K-length prefix itself is a palindrome using the precomputed table.
        # A string is "good" if NO contiguous substring of length K is a palindrome.
        # This includes the first K characters.
        if not is_palindrome_K[state_int]:
             # If the prefix is NOT a palindrome, it's a valid starting state for the DP at index K.
             # Add its count to the corresponding state index in `dp_curr`.
            dp_curr[state_int] = (dp_curr[state_int] + count) % MOD
            
    # Phase 2: Main DP loop. Iterate from index i = K+1 up to N.
    # `dp_curr` holds counts for states ending at position i-1. Each state represents a K-length suffix T[i-K..i-1].
    # Compute `dp_next` for states ending at position i. Each state represents T[i-K+1..i].
    for i in range(K + 1, N + 1):
        # Initialize counts for states ending at position i to zero
        dp_next = [0] * (1 << K) 
        # Get the character constraint from S at the current position i (S is 0-indexed)
        s_char = S[i-1] 
        
        # Iterate through all possible previous states (represented by K-length suffixes ending at i-1)
        # `curr_state_int` is the integer representation of the suffix T[i-K ... i-1].
        # `count` is the number of ways to reach this state.
        for curr_state_int, count in enumerate(dp_curr):
            # Optimization: Skip if count is 0
            if count == 0: continue 
            
            # Consider appending 'A' (represented by bit 0) as character t_i
            if s_char == '?' or s_char == 'A':
                # Calculate the integer representation of the new K-length suffix T[i-K+1 ... i-1 A].
                # This is done by left-shifting `curr_state_int`, adding 0 (for 'A'), and applying the mask.
                new_state_int = ((curr_state_int << 1) | 0) & mask
                
                # Check if this new K-length suffix T[i-K+1..i] is a palindrome using precomputed table.
                if not is_palindrome_K[new_state_int]:
                    # If not a palindrome, this path is valid. Add the count from the previous state
                    # to the count for this new state, taking modulo MOD.
                    dp_next[new_state_int] = (dp_next[new_state_int] + count) % MOD
            
            # Consider appending 'B' (represented by bit 1) as character t_i
            if s_char == '?' or s_char == 'B':
                # Calculate the integer representation of the new K-length suffix T[i-K+1 ... i-1 B].
                new_state_int = ((curr_state_int << 1) | 1) & mask
                
                # Check if this new K-length suffix is a palindrome.
                if not is_palindrome_K[new_state_int]:
                    # If not a palindrome, this path is valid. Add count contribution.
                    dp_next[new_state_int] = (dp_next[new_state_int] + count) % MOD
                    
        # The computed counts in `dp_next` are now the current counts for states ending at position i.
        # Update `dp_curr` to `dp_next` for the next iteration.
        dp_curr = dp_next

    # The final answer is the sum of counts in `dp_curr` after processing up to N.
    # This sums up the number of ways for all valid strings of length N.
    total_count = sum(dp_curr) % MOD
    # Print the final answer.
    print(total_count)

# Execute the solve function to run the program
solve()