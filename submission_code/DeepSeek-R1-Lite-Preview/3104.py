class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums_sorted = sorted(nums)
        count = 0
        
        # Check for k = 0
        if n > 0 and nums_sorted[0] > 0:
            count += 1
        
        # Check for k in 1 to n-1
        for k in range(1, n):
            if nums_sorted[k] > k and nums_sorted[k-1] < k:
                count += 1
        
        # Check for k = n
        if nums_sorted[n-1] < n:
            count += 1
        
        return count