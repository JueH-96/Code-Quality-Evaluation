from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        INF = float('inf')
        n = len(nums)
        m = len(andValues)
        dp = [[INF] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 elements, 0 subarrays
        
        for j in range(1, m + 1):
            current = {}
            for i in range(1, n + 1):
                num = nums[i - 1]
                temp = {}
                
                # Start new subarray with current element
                prev_dp = dp[i - 1][j - 1]
                if prev_dp != INF:
                    if num in temp:
                        if prev_dp < temp[num]:
                            temp[num] = prev_dp
                    else:
                        temp[num] = prev_dp
                
                # Extend previous subarrays by current element
                for and_val in current:
                    new_and = and_val & num
                    prev_val = current[and_val]
                    if new_and in temp:
                        if prev_val < temp[new_and]:
                            temp[new_and] = prev_val
                    else:
                        temp[new_and] = prev_val
                
                current = temp  # Update current to new temp dictionary
                
                # Check if current AND value matches the required andValues[j-1]
                target = andValues[j - 1]
                if target in current:
                    best_prev = current[target]
                    dp[i][j] = min(dp[i][j], best_prev + num)
        
        return dp[n][m] if dp[n][m] != INF else -1