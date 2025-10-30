import collections

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # dp[i] will store the minimum cost to remove elements from original index i to n-1.
        # This implies that nums[i] is always treated as the first element of the current subproblem.
        
        # Initialize dp array with infinity. dp[n] is 0.
        dp = [float('inf')] * (n + 1)
        dp[n] = 0 

        # Fill dp table from right to left
        for i in range(n - 1, -1, -1):
            
            # Base case: 1 element remaining from current `i` (nums[i])
            if i == n - 1:
                dp[i] = nums[i]
                continue
            
            # Base case: 2 elements remaining from current `i` (nums[i], nums[i+1])
            # Operation: remove both. Cost is max of them.
            if i == n - 2:
                dp[i] = max(nums[i], nums[i+1])
                continue

            # General case: 3 or more elements remain from current `i`
            # (nums[i], nums[i+1], nums[i+2], ...)
            
            # Option 1: Remove nums[i] and nums[i+1]
            # Cost for this step: max(nums[i], nums[i+1])
            # Remaining elements start from nums[i+2]. Add dp[i+2].
            cost1 = max(nums[i], nums[i+1]) + dp[i+2]
            dp[i] = min(dp[i], cost1)
            
            # Option 2: Remove nums[i] and nums[i+2]
            # Cost for this step: max(nums[i], nums[i+2])
            # Remaining elements: nums[i+1] (val_kept_op2) followed by nums[i+3:] (suffix_start_idx = i+3).
            val_kept_op2 = nums[i+1]
            
            remaining_len_suffix_op2 = n - (i + 3) # Length of nums[i+3:]
            current_cost_after_op2 = 0
            
            if remaining_len_suffix_op2 == 0:
                # Suffix is empty, only val_kept_op2 remains.
                current_cost_after_op2 = val_kept_op2
            elif remaining_len_suffix_op2 == 1:
                # Suffix has one element (nums[i+3]). val_kept_op2 and nums[i+3] remain.
                current_cost_after_op2 = max(val_kept_op2, nums[i+3])
            else: # remaining_len_suffix_op2 >= 2
                # Suffix has two or more elements. val_kept_op2 is processed, then dp[i+3] for rest.
                current_cost_after_op2 = val_kept_op2 + dp[i+3]
            
            cost2 = max(nums[i], nums[i+2]) + current_cost_after_op2
            dp[i] = min(dp[i], cost2)

            # Option 3: Remove nums[i+1] and nums[i+2]
            # Cost for this step: max(nums[i+1], nums[i+2])
            # Remaining elements: nums[i] (val_kept_op3) followed by nums[i+3:] (suffix_start_idx = i+3).
            val_kept_op3 = nums[i]

            remaining_len_suffix_op3 = n - (i + 3)
            current_cost_after_op3 = 0

            if remaining_len_suffix_op3 == 0:
                current_cost_after_op3 = val_kept_op3
            elif remaining_len_suffix_op3 == 1:
                current_cost_after_op3 = max(val_kept_op3, nums[i+3])
            else: # remaining_len_suffix_op3 >= 2
                current_cost_after_op3 = val_kept_op3 + dp[i+3]

            cost3 = max(nums[i+1], nums[i+2]) + current_cost_after_op3
            dp[i] = min(dp[i], cost3)
            
        return dp[0]