class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        m = n // 2  # 0-based index for median
        
        operations = 0
        
        if nums_sorted[m] > k:
            # Decrease the m-th element to k
            operations += nums_sorted[m] - k
            # Decrease elements before m that are greater than k
            for i in range(m):
                if nums_sorted[i] > k:
                    operations += nums_sorted[i] - k
        elif nums_sorted[m] < k:
            # Increase the m-th element to k
            operations += k - nums_sorted[m]
            # Increase elements after m that are less than k
            for i in range(m + 1, n):
                if nums_sorted[i] < k:
                    operations += k - nums_sorted[i]
        # If nums_sorted[m] == k, no operations needed
        
        return operations