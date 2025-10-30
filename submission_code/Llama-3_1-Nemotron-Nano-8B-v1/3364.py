from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        m = len(andValues)
        n = len(nums)
        if m > n:
            return -1
        
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0  # Base case: 0 splits, 0 elements
        
        # Handle k=1 separately using prefix AND array
        if m >= 1:
            prefix_and = [0] * n
            prefix_and[0] = nums[0]
            for i in range(1, n):
                prefix_and[i] = prefix_and[i-1] & nums[i]
            
            for i in range(1, n + 1):
                if i == 1:
                    current_and = nums[0]
                else:
                    current_and = prefix_and[i-2] & nums[i-1] if i-2 >= 0 else nums[i-1]
                if current_and == andValues[0]:
                    dp[1][i] = nums[i-1]
        
        # Handle k >= 2
        for k in range(2, m + 1):
            prev_list = {}  # current_and: min_prev_sum
            for i in range(1, n + 1):
                current_num = nums[i-1]
                new_list = {}
                
                # Process previous list to update AND values
                for and_val, prev_sum in prev_list.items():
                    new_and = and_val & current_num
                    if new_and in new_list:
                        if prev_sum < new_list[new_and]:
                            new_list[new_and] = prev_sum
                    else:
                        new_list[new_and] = prev_sum
                
                # Add new subarray starting at i (current_num)
                if dp[k-1][i-1] != INF:
                    new_and = current_num
                    new_sum = dp[k-1][i-1] + current_num
                    if new_and in new_list:
                        if new_sum < new_list[new_and]:
                            new_list[new_and] = new_sum
                    else:
                        new_list[new_and] = new_sum
                
                # Check if current k's target AND is present
                target_and = andValues[k-1]
                if target_and in new_list:
                    candidate = new_list[target_and] + current_num
                    if candidate < dp[k][i]:
                        dp[k][i] = candidate
                
                prev_list = new_list.copy()
        
        return dp[m][n] if dp[m][n] != INF else -1