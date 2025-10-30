class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # dp[x] represents the maximum length of consecutive sequence ending at value x
        dp = defaultdict(int)
        
        # Sort the array to process elements in order
        nums.sort()
        
        for num in nums:
            # For each number, we have two choices:
            # 1. Use num (don't increase)
            # 2. Use num+1 (increase by 1)
            
            # If we use num, it can extend a sequence ending at num-1
            new_dp_num = dp[num - 1] + 1
            
            # If we use num+1, it can extend a sequence ending at num
            new_dp_num_plus_1 = dp[num] + 1
            
            # Update dp values
            dp[num] = max(dp[num], new_dp_num)
            dp[num + 1] = max(dp[num + 1], new_dp_num_plus_1)
        
        return max(dp.values())