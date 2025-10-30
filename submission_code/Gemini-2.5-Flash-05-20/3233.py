from collections import Counter

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)

        # N_O_A[i][char_code]: stores the index of the first occurrence of char_code in s[i:]
        # If not found, it stores N.
        N_O_A = [[n] * 26 for _ in range(n + 1)]

        for i in range(n - 1, -1, -1):
            for c_code in range(26):
                N_O_A[i][c_code] = N_O_A[i + 1][c_code]
            N_O_A[i][ord(s[i]) - ord('a')] = i

        # Helper function to get the end index of the longest prefix satisfying the k-distinct char constraint.
        # This function simulates finding the (k+1)-th distinct character's first occurrence.
        # `is_modified_char_at_idx`: True if s[modified_idx] is considered `new_char`
        def get_next_segment_end(start_idx: int, k: int, is_modified_char_at_idx: bool, 
                                 modified_idx: int, new_char_code: int) -> int:
            
            indices = [] # Stores the first occurrence index for each distinct character in the potential segment.

            old_char_code = -1
            if is_modified_char_at_idx and start_idx <= modified_idx < n:
                old_char_code = ord(s[modified_idx]) - ord('a')

            for c_code in range(26):
                pos = N_O_A[start_idx][c_code] # Original first occurrence of char `c_code` from `start_idx`.

                if is_modified_char_at_idx and start_idx <= modified_idx < n:
                    if c_code == old_char_code and pos == modified_idx:
                        # The character at modified_idx was `old_char_code` and it was its first occurrence from `start_idx`.
                        # Now it's changed, so the actual first occurrence of `old_char_code` from `start_idx` is the one after `modified_idx`.
                        pos = N_O_A[modified_idx + 1][c_code]
                    elif c_code == new_char_code:
                        # `new_char_code` is now present at `modified_idx`.
                        # Its first occurrence from `start_idx` is either its original one or `modified_idx` itself, whichever is earlier.
                        pos = min(pos, modified_idx)
                
                if pos < n: # If character found within string bounds
                    indices.append(pos)
            
            # Sort to find the k-th smallest (i.e., (k+1)-th distinct char)
            indices.sort() 
            
            if len(indices) <= k:
                # If there are k or fewer distinct characters, the segment can extend to the end of the string.
                return n 
            else:
                # The k-th smallest index is where the (k+1)-th distinct char first appears, forcing the cut.
                return indices[k] 

        # dp[i][0]: max partitions from s[i:] with no changes allowed in s[i:].
        # dp[i][1]: max partitions from s[i:] with one change allowed anywhere in s[i:].
        dp = [[0] * 2 for _ in range(n + 1)]

        for i in range(n - 1, -1, -1):
            # Calculate dp[i][0]
            # Next segment ends based on original string properties
            next_end_0 = get_next_segment_end(i, k, False, -1, -1)
            dp[i][0] = 1 + dp[next_end_0][0]

            # Calculate dp[i][1]
            max_val_for_dp1 = 0

            # Option A: Don't use the change in the current segment [i, next_end_0-1]
            # The current partition is based on original string. One change is still available for the rest.
            max_val_for_dp1 = max(max_val_for_dp1, 1 + dp[next_end_0][1])

            # Option B: Use the change at index `i` (change s[i] to any new_char)
            # The current partition is based on the modified s[i]. No changes available for the rest.
            for new_c_code in range(26):
                # Calculate next segment end if s[i] is changed to new_c_code
                next_end_1_modified_at_i = get_next_segment_end(i, k, True, i, new_c_code)
                max_val_for_dp1 = max(max_val_for_dp1, 1 + dp[next_end_1_modified_at_i][0])
            
            dp[i][1] = max_val_for_dp1

        # The final answer is the maximum partitions starting from index 0, with one change allowed.
        return dp[0][1]