from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # best_diff[j] will store the maximum (nums[i] - nums[j]) for i < j
        best_diff = [float('-inf')] * n
        prefix_max = float('-inf')  # Tracks the maximum nums[i] for i < j
        
        for j in range(n):
            if prefix_max != float('-inf'):
                best_diff[j] = prefix_max - nums[j]
            prefix_max = max(prefix_max, nums[j])
        
        # suffix_max[k] will store the maximum nums[x] for x >= k
        suffix_max = [0] * n
        suffix_max[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(suffix_max[i + 1], nums[i])
        
        # Compute the maximum value of best_diff[j] * suffix_max[j + 1] for j < n - 1
        ans = 0
        for j in range(n - 1):
            if best_diff[j] != float('-inf'):
                val = best_diff[j] * suffix_max[j + 1]
                ans = max(ans, val)
        
        return max(ans, 0)