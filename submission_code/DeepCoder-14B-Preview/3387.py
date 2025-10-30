class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        m = len(nums_sorted) // 2
        cost = 0
        
        # Adjust elements from the start up to the median position
        for i in range(m + 1):
            if nums_sorted[i] > k:
                cost += nums_sorted[i] - k
        
        # Adjust elements from the median position to the end
        for i in range(m, len(nums_sorted)):
            if nums_sorted[i] < k:
                cost += k - nums_sorted[i]
        
        return cost