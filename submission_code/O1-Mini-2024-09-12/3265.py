from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        min_prefix_sum = {}
        max_sum = float('-inf')
        
        for num in nums:
            # Check if there's a previous number that is num + k
            if num + k in min_prefix_sum:
                potential_sum = prefix_sum + num - min_prefix_sum[num + k]
                if potential_sum > max_sum:
                    max_sum = potential_sum
            # Check if there's a previous number that is num - k
            if num - k in min_prefix_sum:
                potential_sum = prefix_sum + num - min_prefix_sum[num - k]
                if potential_sum > max_sum:
                    max_sum = potential_sum
            # Update the minimum prefix sum for the current number
            if num in min_prefix_sum:
                min_prefix_sum[num] = min(min_prefix_sum[num], prefix_sum)
            else:
                min_prefix_sum[num] = prefix_sum
            # Update the running prefix sum
            prefix_sum += num
        return max_sum if max_sum != float('-inf') else 0