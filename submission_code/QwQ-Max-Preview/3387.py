class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        n = len(nums)
        m = n // 2
        operations = abs(nums_sorted[m] - k)
        
        # Adjust elements before m to be <= k
        for i in range(m):
            if nums_sorted[i] > k:
                operations += nums_sorted[i] - k
        
        # Adjust elements after m to be >= k
        for i in range(m + 1, n):
            if nums_sorted[i] < k:
                operations += k - nums_sorted[i]
        
        return operations