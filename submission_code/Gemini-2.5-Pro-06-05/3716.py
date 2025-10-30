class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        
        n = len(nums)
        # Constraints state 2 <= nums.length, so n is at least 2.

        MAX_VAL = 300
        MAX_DIFF = 299
        
        # dp[v][d]: length of the longest subsequence ending with value v,
        # with the absolute difference to the previous element being d.
        dp = [[0] * (MAX_DIFF + 1) for _ in range(MAX_VAL + 1)]
        
        # suffix_max_dp[v][d]: max(dp[v][k] for k >= d).
        # This is an optimization for O(1) lookup.
        suffix_max_dp = [[0] * (MAX_DIFF + 1) for _ in range(MAX_VAL + 1)]
        
        # max_len_at_val[v]: max length of any valid subsequence ending at value v.
        # This is to quickly check if a value 'v' has been seen before.
        max_len_at_val = [0] * (MAX_VAL + 1)
        
        max_overall_len = 1

        for u in nums:
            # This will store the potential new lengths for subsequences ending in u.
            dp_u_row = [0] * (MAX_DIFF + 1)
            
            for v in range(1, MAX_VAL + 1):
                # If v has not been seen before, it cannot be a predecessor.
                if max_len_at_val[v] > 0:
                    d = abs(u - v)
                    
                    # Case 1: Extend a subsequence of length 1, i.e., [v].
                    # The new subsequence is [v, u], length 2.
                    candidate_len = 2
                    
                    # Case 2: Extend a longer subsequence ending in v.
                    max_prev_len = suffix_max_dp[v][d]
                    if max_prev_len > 0:
                        candidate_len = max(candidate_len, 1 + max_prev_len)
                    
                    if candidate_len > dp_u_row[d]:
                        dp_u_row[d] = candidate_len

            # Update the main DP tables with the new information about u.
            needs_suffix_recompute = False
            for d in range(MAX_DIFF + 1):
                if dp_u_row[d] > dp[u][d]:
                    dp[u][d] = dp_u_row[d]
                    needs_suffix_recompute = True
            
            if needs_suffix_recompute:
                suffix_max_dp[u][MAX_DIFF] = dp[u][MAX_DIFF]
                for d in range(MAX_DIFF - 1, -1, -1):
                    suffix_max_dp[u][d] = max(dp[u][d], suffix_max_dp[u][d+1])
            
            # Mark u as seen and update its max length
            max_len_at_val[u] = max(max_len_at_val[u], 1)
            current_u_max_len = suffix_max_dp[u][0]
            max_len_at_val[u] = max(max_len_at_val[u], current_u_max_len)

            # Update the overall maximum length.
            max_overall_len = max(max_overall_len, max_len_at_val[u])
            
        return max_overall_len