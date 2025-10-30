import collections

class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1

        # Calculate the length of the resulting string.
        # If str2 is empty, it's an invalid scenario per constraints (m >= 1).

        word_arr = [None] * L # Use a list for mutable characters

        # Step 1: Apply 'T' constraints and check for immediate conflicts
        # Any character fixed by a 'T' constraint (str1[i] == 'T') must be str2[j]
        for i in range(n):
            if str1[i] == 'T':
                for j in range(m):
                    k = i + j # Index in the target word_arr
                    # k will always be within [0, L-1] based on L's definition
                    # (max i = n-1, max j = m-1 -> max k = n-1+m-1 = L-1)
                    if word_arr[k] is None:
                        word_arr[k] = str2[j]
                    elif word_arr[k] != str2[j]:
                        # Conflict: A character is forced to be different values by overlapping 'T' constraints.
                        return "" # No solution

        # Step 2: Precompute all_future_T_match_for_F for 'F' constraints
        # all_future_T_match_for_F[i][j] is True if the segment word_arr[i+j : i+m]
        # (of length m-j) consists entirely of characters fixed by 'T' constraints
        # AND these characters match str2[j : m].
        # This helps determine if a choice for word_arr[k] would force an 'F' constraint to fail later.
        
        # Initialize with False
        all_future_T_match_for_F = [[False] * m for _ in range(n)]

        for i in range(n):
            if str1[i] == 'F':
                # Iterate j from m-1 down to 0 (right to left within the window)
                for j in range(m - 1, -1, -1):
                    k_idx = i + j # Corresponding index in word_arr
                    
                    # If this part of the window is out of bounds (shouldn't happen for valid j)
                    # or word_arr[k_idx] is not fixed by 'T' (is None)
                    # or word_arr[k_idx] is fixed by 'T' but doesn't match str2[j]
                    if k_idx >= L or word_arr[k_idx] is None or word_arr[k_idx] != str2[j]:
                        all_future_T_match_for_F[i][j] = False
                    else: # word_arr[k_idx] is fixed by 'T' and matches str2[j]
                        if j == m - 1: # This is the last character in the current m-length window
                            all_future_T_match_for_F[i][j] = True
                        else: # Check if the remainder of the segment also matches
                            all_future_T_match_for_F[i][j] = all_future_T_match_for_F[i][j + 1]

                # Critical check: If the entire 'F' window (str1[i]) is forced by 'T' constraints
                # to match str2, then there's a conflict.
                if all_future_T_match_for_F[i][0]:
                    return "" # Conflict: 'F' constraint requires a mismatch, but all fixed characters match str2.

        # Step 3: Initialize f_match_state for 'F' constraints
        # f_match_state[i]:
        #   -1 if str1[i] == 'F' is already satisfied (due to a T-fixed mismatch within its window).
        #   Otherwise, it stores `current_match_length`, the length of the prefix of str2
        #   that matches word_arr starting from index i, considering only T-fixed characters.
        f_match_state = [-1] * n # Initialize all to -1 (satisfied/not-applicable)

        for i in range(n):
            if str1[i] == 'F':
                found_mismatch_by_T = False
                current_match_len = 0
                for j in range(m):
                    k = i + j
                    if word_arr[k] is not None: # If fixed by a 'T' constraint
                        if word_arr[k] != str2[j]: # Check for mismatch
                            found_mismatch_by_T = True
                            break # Mismatch found, 'F' constraint is satisfied
                        else: # Character matches
                            current_match_len += 1
                    else: # First None encountered, no more T-fixed matches for this prefix
                        break
                
                if found_mismatch_by_T:
                    f_match_state[i] = -1 # 'F' constraint satisfied by a T-fixed mismatch
                else:
                    f_match_state[i] = current_match_len # Still actively matching a prefix of str2


        # Step 4: Greedy fill word_arr from left to right (k = 0 to L-1)
        for k in range(L):
            if word_arr[k] is not None: # Character is already fixed by a 'T' constraint
                char_k = word_arr[k]
            else: # Character is not fixed, choose the lexicographically smallest 'c'
                char_k = None # Will try to find a valid character
                for c_ord in range(ord('a'), ord('z') + 1):
                    c = chr(c_ord)
                    is_valid_choice = True
                    
                    # Check all 'F' constraints that overlap with current position k.
                    # An 'F' constraint str1[i] affects word_arr[k] if i <= k < i + m.
                    # This means i is in the range [max(0, k - m + 1), min(n - 1, k)].
                    for i in range(max(0, k - m + 1), min(n, k + 1)):
                        if str1[i] == 'F' and f_match_state[i] != -1: # Active 'F' constraint
                            # If word_arr[i : k] (prefix for this F-window) currently matches str2[0 : k-i]
                            if f_match_state[i] == k - i: 
                                if c == str2[k - i]: # If proposed char 'c' would continue the match
                                    # This 'F' constraint would then fail if:
                                    # 1. This is the last character of the window (k == i + m - 1), 
                                    #    meaning the entire window would now match str2.
                                    # 2. OR, if the remaining characters (word_arr[k+1 : i+m]) are ALL fixed by 'T' 
                                    #    and also match str2 (checked using all_future_T_match_for_F).
                                    if (k == i + m - 1) or \
                                       (k + 1 < i + m and all_future_T_match_for_F[i][k - i + 1]):
                                        is_valid_choice = False
                                        break # This 'c' is not valid for word_arr[k], try next char
                    
                    if is_valid_choice:
                        char_k = c
                        break # Found the lexicographically smallest valid character for word_arr[k]
                
                if char_k is None:
                    # No valid character found for word_arr[k] after trying 'a' through 'z'
                    return "" # No solution possible

            word_arr[k] = char_k # Set the chosen character for word_arr[k]

            # Update f_match_state for all 'F' constraints that are affected by word_arr[k]
            # (i.e., str1[i] where i <= k < i + m and f_match_state[i] is still active).
            for i in range(max(0, k - m + 1), min(n, k + 1)):
                if str1[i] == 'F' and f_match_state[i] != -1: # Active 'F' constraint
                    # If this 'F' constraint was actively matching up to k-1 (i.e., f_match_state[i] == k-i)
                    if f_match_state[i] == k - i:
                        if char_k == str2[k - i]: # If char_k continued the match
                            f_match_state[i] += 1 # Increment matched length
                        else: # char_k created a mismatch
                            f_match_state[i] = -1 # 'F' constraint is now satisfied
                    # If f_match_state[i] < k-i, it means a mismatch was already found for `i` before `k`,
                    # so its f_match_state[i] should already be -1. No action needed.

        return "".join(word_arr) # Return the constructed string