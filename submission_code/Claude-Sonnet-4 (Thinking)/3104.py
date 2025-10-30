class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        
        # Check k = 0 (no students selected)
        if nums[0] > 0:
            count += 1
        
        # Check k = 1 to n-1 (some students selected)
        for k in range(1, n):
            if nums[k-1] < k and nums[k] > k:
                count += 1
        
        # Check k = n (all students selected, always valid given constraints)
        count += 1
        
        return count