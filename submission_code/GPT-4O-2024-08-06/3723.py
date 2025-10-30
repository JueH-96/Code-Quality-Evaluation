class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        good_sum = 0
        
        for i in range(n):
            is_good = True
            
            # Check the element at index i - k
            if i - k >= 0 and nums[i] <= nums[i - k]:
                is_good = False
            
            # Check the element at index i + k
            if i + k < n and nums[i] <= nums[i + k]:
                is_good = False
            
            # If the current number is good, add it to the sum
            if is_good:
                good_sum += nums[i]
        
        return good_sum