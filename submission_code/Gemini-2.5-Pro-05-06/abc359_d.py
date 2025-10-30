import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S_chars = sys.stdin.readline().strip()
    MOD = 998244353

    # Precompute K-bit masks that are palindromes
    # A=0, B=1
    is_palindrome_lookup = [False] * (1 << K)
    if K > 0: # K >= 2 as per constraints, so K is positive
        for k_mask_val in range(1 << K):
            is_pal = True
            # Check bits from outside in
            for j in range(K // 2): # e.g. K=4, j=0,1. K=5, j=0,1 (center bit doesn't need explicit check)
                bit_low = (k_mask_val >> j) & 1            # j-th bit from right (0-indexed)
                bit_high = (k_mask_val >> (K - 1 - j)) & 1 # (K-1-j)-th bit from right (corresponding bit)
                if bit_low != bit_high:
                    is_pal = False
                    break
            if is_pal:
                is_palindrome_lookup[k_mask_val] = True
    
    # dp_curr: dictionary mapping mask to count.
    # After processing S_chars[0...i-1] (string of length i):
    # mask represents the suffix S_chars[max(0, i-(K-1)) ... i-1].
    # Length of this mask is L_mask = min(i, K-1).
    dp_curr = {0: 1} # Base case: For prefix of length 0, L_mask=0. Mask 0 represents empty string, 1 way.

    for i in range(N): # Current character S_chars[i] to fill (0-indexed). Forms string S_chars[0...i].
        dp_next = {} # Stores DP states for strings of length i+1.
        
        # L_prev_mask is the length of masks in dp_curr (which are for strings of length i).
        L_prev_mask = min(i, K - 1)

        possible_char_values = []
        if S_chars[i] == 'A':
            possible_char_values.append(0) # 0 for 'A'
        elif S_chars[i] == 'B':
            possible_char_values.append(1) # 1 for 'B'
        else: # S_chars[i] == '?'
            possible_char_values.append(0) # Try 'A'
            possible_char_values.append(1) # Try 'B'

        for prev_mask, count in dp_curr.items():
            for char_val in possible_char_values:
                # String S_chars[0...i] is being formed. Its current length is i+1.
                
                # Palindrome Check:
                # A K-length suffix S_chars[i-K+1 ... i] is formed if current_length (i+1) >= K.
                # This is equivalent to i >= K-1.
                if i >= K - 1:
                    # If i >= K-1, then L_prev_mask = min(i, K-1) = K-1.
                    # So prev_mask has K-1 bits. It represents S_chars[i-(K-1) ... i-1].
                    # Appending char_val (for S_chars[i]) forms S_chars[i-(K-1) ... i].
                    # This string has K characters. Its K-bit integer representation is k_mask_to_check.
                    k_mask_to_check = (prev_mask << 1 | char_val) 
                    
                    if is_palindrome_lookup[k_mask_to_check]:
                        continue # This path creates a K-length palindrome, so it's invalid.
                
                # If no K-length palindrome formed (or string not long enough to form one ending at S_chars[i]):
                # Determine new_mask for dp_next.
                # dp_next stores masks for strings of length i+1.
                # The length of these new masks is L_new_mask = min(i+1, K-1).
                L_new_mask = min(i + 1, K - 1)
                
                # The string that new_mask represents is the suffix of S_chars[0...i] of length L_new_mask.
                # This suffix is S_chars[(i+1)-L_new_mask ... i].
                # It can be obtained by taking the last L_new_mask bits of the string
                # S_chars[max(0, i-L_prev_mask) ... i].
                # This "potentially longer" string S_chars[max(0, i-L_prev_mask) ... i] is represented by (prev_mask << 1 | char_val).
                # It has L_prev_mask + 1 bits.
                temp_extended_mask = (prev_mask << 1 | char_val)
                # Truncate to L_new_mask bits to get the actual_new_mask.
                actual_new_mask = temp_extended_mask & ((1 << L_new_mask) - 1)
                
                dp_next[actual_new_mask] = (dp_next.get(actual_new_mask, 0) + count) % MOD
        
        dp_curr = dp_next # Move to next length

    # After N iterations, dp_curr contains states for strings of length N.
    # Sum of all counts is the total number of good strings.
    total_good_strings = 0
    for count in dp_curr.values():
        total_good_strings = (total_good_strings + count) % MOD
    
    print(total_good_strings)

solve()