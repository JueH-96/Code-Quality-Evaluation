class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        m = n // 2  # Index of the median for 0-based indexing (larger of middle two if even)
        
        # If the current median is already k, no operations needed
        if nums[m] == k:
            return 0
        
        # If the current median is less than k, we need to "push up" all elements
        # from index m to the end so that they are at least k.
        if nums[m] < k:
            ops = 0
            for i in range(m, n):
                if nums[i] < k:
                    ops += (k - nums[i])
            return ops
        
        # Otherwise (nums[m] > k), we need to "push down" elements from index 0 to m
        # so that they are at most k.
        ops = 0
        for i in range(m + 1):
            if nums[i] > k:
                ops += (nums[i] - k)
        return ops