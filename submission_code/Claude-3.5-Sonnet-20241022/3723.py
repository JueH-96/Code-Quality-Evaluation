class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        good_sum = 0
        
        for i in range(n):
            is_good = True
            
            # Check left element (i - k) if it exists
            if i - k >= 0:
                if nums[i] <= nums[i - k]:
                    is_good = False
            
            # Check right element (i + k) if it exists
            if i + k < n:
                if nums[i] <= nums[i + k]:
                    is_good = False
            
            # If the number is good, add it to the sum
            if is_good:
                good_sum += nums[i]
        
        return good_sum