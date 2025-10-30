class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        
        for i in range(n):
            is_good = True
            
            # Check if nums[i] > nums[i-k] (if i-k is valid)
            if i - k >= 0 and nums[i] <= nums[i - k]:
                is_good = False
            
            # Check if nums[i] > nums[i+k] (if i+k is valid)
            if i + k < n and nums[i] <= nums[i + k]:
                is_good = False
            
            if is_good:
                total += nums[i]
        
        return total