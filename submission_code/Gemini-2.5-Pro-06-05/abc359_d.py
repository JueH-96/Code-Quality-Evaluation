# YOUR CODE HERE
import sys

def main():
    """
    Solves the problem using dynamic programming with bitmasking.
    """
    # Read problem inputs from stdin
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        N, K = map(int, line1.split())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle cases where input might be empty.
        return

    # Constants
    MOD = 998244353
    K1 = K - 1

    # Number of possible binary strings (masks) of length K-1 and K
    MASK_LEN_K1 = 1 << K1
    MASK_LEN_K = 1 << K

    # Precompute which binary strings of length K are palindromes.
    # A mask represents a binary string. We use LSB-first encoding, where
    # bit `i` corresponds to the character at index `i` of the string.
    # e.g., for K=4, string "ABBA" (A=0, B=1) is represented by mask 0110_2.
    is_palindrome_K = [False] * MASK_LEN_K
    for mask in range(MASK_LEN_K):
        is_pal = True
        for i in range(K // 2):
            left_bit = (mask >> i) & 1
            right_bit = (mask >> (K - 1 - i)) & 1
            if left_bit != right_bit:
                is_pal = False
                break
        if is_pal:
            is_palindrome_K[mask] = True

    # dp[mask] stores the number of valid ways to form a prefix
    # such that its suffix of length K-1 is represented by 'mask'.
    dp = [0] * MASK_LEN_K1

    # --- Base Case ---
    # Initialize DP for prefixes of length K-1.
    # The mask represents the full prefix T[0...K-2].
    if K1 > 0:
        for mask in range(MASK_LEN_K1):
            is_consistent = True
            # Check if the string represented by 'mask' is consistent with S[0...K-2]
            for j in range(K1):
                bit = (mask >> j) & 1
                char = 'B' if bit else 'A'
                if S[j] != '?' and S[j] != char:
                    is_consistent = False
                    break
            if is_consistent:
                dp[mask] = 1
    else: # K=1. A single-character string is always a palindrome.
          # Constraints state 2 <= K, so this branch is not strictly needed.
        print(0)
        return

    # --- DP Iterations ---
    # Extend prefixes from length i-1 to i, for i from K to N.
    for i in range(K, N + 1):
        new_dp = [0] * MASK_LEN_K1
        for prev_mask in range(MASK_LEN_K1):
            if dp[prev_mask] == 0:
                continue
            
            # Try appending 'A' (c_val=0) and 'B' (c_val=1)
            for c_val in range(2):
                char = 'B' if c_val else 'A'
                
                # The new character must be consistent with S[i-1]
                if S[i-1] != '?' and S[i-1] != char:
                    continue

                # A new substring of length K is formed: T[i-K...i-1].
                # Its mask is constructed from prev_mask (for T[i-K...i-2])
                # and c_val (for T[i-1]).
                # With LSB-first encoding, T[i-1] corresponds to bit K1.
                k_string_mask = prev_mask | (c_val << K1)
                
                # If this new substring is a palindrome, this path is invalid.
                if is_palindrome_K[k_string_mask]:
                    continue
                
                # If valid, update the DP table for the next state.
                # The new suffix of length K-1 is T[i-K+1...i-1].
                # This is formed by right-shifting prev_mask (discards T[i-K])
                # and adding the new char T[i-1] at the MSB position.
                new_mask = (prev_mask >> 1) | (c_val << (K1 - 1))
                new_dp[new_mask] = (new_dp[new_mask] + dp[prev_mask]) % MOD
        
        # The new DP state becomes the current state for the next iteration.
        dp = new_dp

    # The total number of good strings is the sum of counts for all possible suffixes.
    total_count = sum(dp) % MOD
    print(total_count)

if __name__ == "__main__":
    main()