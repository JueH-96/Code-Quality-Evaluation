class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        
        for i in range(n):
            good = True
            
            # Check left neighbor (i - k)
            if i - k >= 0:
                good = good and nums[i] > nums[i - k]
            
            # Check right neighbor (i + k) 
            if i + k < n:
                good = good and nums[i] > nums[i + k]
            
            if good:
                total += nums[i]
        
        return total