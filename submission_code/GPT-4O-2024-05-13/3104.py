class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        
        # Check the case where no student is selected
        if nums[0] > 0:
            count += 1
        
        # Check the cases where some students are selected
        for i in range(n):
            if i + 1 > nums[i] and (i == n - 1 or i + 1 < nums[i + 1]):
                count += 1
        
        return count