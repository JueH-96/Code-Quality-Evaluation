class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        good_sum = 0
        n = len(nums)
        
        for i in range(n):
            is_good = True
            
            # Check the left index
            if i - k >= 0 and nums[i] <= nums[i - k]:
                is_good = False
            
            # Check the right index
            if i + k < n and nums[i] <= nums[i + k]:
                is_good = False
            
            if is_good:
                good_sum += nums[i]
        
        return good_sum