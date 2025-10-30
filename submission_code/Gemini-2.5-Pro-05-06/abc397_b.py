import math

def solve():
    S = input()
    N = len(S)

    # prev_dp_even stores the minimum length of an alternating i-o string T_prefix
    # such that S[0...k-1] (characters processed so far) is a subsequence of T_prefix,
    # T_prefix has an even length, and S[k-1] (if k > 0) corresponds to T_prefix's last character.
    # prev_dp_odd is similar, but for T_prefix having an odd length.
    # k represents the count of characters processed from S.

    # Base case (k=0): No characters from S processed yet.
    # To form an even length T_prefix (e.g., an empty string ""), its length is 0.
    prev_dp_even = 0
    
    # To form an odd length T_prefix (e.g., "i" by inserting 'i'), its length is 1.
    prev_dp_odd = 1 

    # Iterate through S, processing S[i] (0-indexed)
    # In each iteration, prev_dp_even/odd are states after processing S[0...i-1]
    # and curr_dp_even/odd will be states after processing S[0...i]
    for char_idx in range(N):
        char_s = S[char_idx]
        
        curr_dp_even = math.inf
        curr_dp_odd = math.inf
        
        if char_s == 'i':
            # Current char S[char_idx] is 'i'.
            # It must match an 'i' in T, which is at an odd position.
            # So, if S[char_idx] is the last character of T matched from S, T will have odd length.
            
            # Path 1: T_prefix for S[0...char_idx-1] was even length (prev_dp_even).
            # The next character required in T is T_{prev_dp_even+1} = 'i'.
            # S[char_idx] ('i') matches. New T_prefix length = prev_dp_even + 1. This T_prefix is odd.
            if prev_dp_even != math.inf: # Check if prev_dp_even is a reachable state
                curr_dp_odd = min(curr_dp_odd, prev_dp_even + 1)
            
            # Path 2: T_prefix for S[0...char_idx-1] was odd length (prev_dp_odd).
            # The next character required in T is T_{prev_dp_odd+1} = 'o'.
            # S[char_idx] ('i') mismatches 'o'. So, insert 'o'. 
            # T_prefix's length becomes prev_dp_odd + 1 (now even).
            # The next character required in T is T_{(prev_dp_odd+1)+1} = 'i'.
            # S[char_idx] ('i') matches. T_prefix's length becomes (prev_dp_odd + 1) + 1. This T_prefix is odd.
            if prev_dp_odd != math.inf: # Check if prev_dp_odd is a reachable state
                curr_dp_odd = min(curr_dp_odd, prev_dp_odd + 2)
                
        elif char_s == 'o':
            # Current char S[char_idx] is 'o'.
            # It must match an 'o' in T, which is at an even position.
            # So, if S[char_idx] is the last character of T matched from S, T will have even length.

            # Path 1: T_prefix for S[0...char_idx-1] was odd length (prev_dp_odd).
            # The next character required in T is T_{prev_dp_odd+1} = 'o'.
            # S[char_idx] ('o') matches. New T_prefix length = prev_dp_odd + 1. This T_prefix is even.
            if prev_dp_odd != math.inf: # Check if prev_dp_odd is a reachable state
                curr_dp_even = min(curr_dp_even, prev_dp_odd + 1)
            
            # Path 2: T_prefix for S[0...char_idx-1] was even length (prev_dp_even).
            # The next character required in T is T_{prev_dp_even+1} = 'i'.
            # S[char_idx] ('o') mismatches 'i'. So, insert 'i'.
            # T_prefix's length becomes prev_dp_even + 1 (now odd).
            # The next character required in T is T_{(prev_dp_even+1)+1} = 'o'.
            # S[char_idx] ('o') matches. T_prefix's length becomes (prev_dp_even + 1) + 1. This T_prefix is even.
            if prev_dp_even != math.inf: # Check if prev_dp_even is a reachable state
                curr_dp_even = min(curr_dp_even, prev_dp_even + 2)
        
        # Update states for the next iteration (after processing S[char_idx])
        prev_dp_even = curr_dp_even
        prev_dp_odd = curr_dp_odd

    # After processing all N characters of S:
    # prev_dp_even is min length of an even T, where S is its subsequence and S[N-1] is matched to T's last char.
    # prev_dp_odd is min length of an odd T, where S is its subsequence and S[N-1] is matched to T's last char.
    
    # The problem requires the final string T to have even length.
    
    # Option 1: T derived from S was already even (length prev_dp_even).
    # This is a candidate for the final T length.
    final_T_length = prev_dp_even
    
    # Option 2: T derived from S was odd (length prev_dp_odd).
    # To make it even, we must append one more character ('o').
    # The new length would be prev_dp_odd + 1.
    if prev_dp_odd != math.inf: # Check if an odd length T was possible
        final_T_length = min(final_T_length, prev_dp_odd + 1)
        
    # The number of insertions is the final T length minus N (original length of S).
    insertions = final_T_length - N
    print(insertions)

solve()