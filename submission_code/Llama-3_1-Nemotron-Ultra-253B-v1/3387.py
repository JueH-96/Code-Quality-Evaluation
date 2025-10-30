from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        m = len(nums_sorted) // 2
        cost = 0
        
        # Change the median element to k
        cost += abs(k - nums_sorted[m])
        
        # Ensure elements after m are >= k
        for num in nums_sorted[m+1:]:
            if num < k:
                cost += (k - num)
        
        # Ensure elements before m are <= k
        for num in nums_sorted[:m]:
            if num > k:
                cost += (num - k)
        
        return cost