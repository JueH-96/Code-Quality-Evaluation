from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = float('inf')
        dp = [[INF] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for k in range(1, m + 1):
            prev_ands = None
            for i in range(n):
                current_ands = {}
                # Process previous_ands (from i-1)
                if prev_ands is not None:
                    for and_val in prev_ands:
                        new_and = and_val & nums[i]
                        min_prev = prev_ands[and_val]
                        if new_and in current_ands:
                            if min_prev < current_ands[new_and]:
                                current_ands[new_and] = min_prev
                        else:
                            current_ands[new_and] = min_prev
                # Add new subarray starting at i
                start_and = nums[i]
                start_prev = dp[i][k-1]
                if start_and in current_ands:
                    if start_prev < current_ands[start_and]:
                        current_ands[start_and] = start_prev
                else:
                    current_ands[start_and] = start_prev
                
                # Check if target is in current_ands
                target = andValues[k-1]
                if target in current_ands:
                    candidate = current_ands[target] + nums[i]
                    if candidate < dp[i+1][k]:
                        dp[i+1][k] = candidate
                # Update prev_ands for next iteration
                prev_ands = current_ands
        
        result = dp[n][m]
        return result if result != INF else -1