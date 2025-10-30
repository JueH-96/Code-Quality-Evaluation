class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        good_sum = 0
        
        for i in range(len(nums)):
            is_good = True
            
            # Check if current element is greater than the element at index i-k (if it exists)
            if i - k >= 0 and nums[i] <= nums[i - k]:
                is_good = False
            
            # Check if current element is greater than the element at index i+k (if it exists)
            if i + k < len(nums) and nums[i] <= nums[i + k]:
                is_good = False
            
            # If both conditions are satisfied (or the indices don't exist), add to sum
            if is_good:
                good_sum += nums[i]
        
        return good_sum