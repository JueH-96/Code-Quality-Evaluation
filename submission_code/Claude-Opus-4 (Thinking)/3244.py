class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        
        # Check if we can create a value smaller than min_val
        for x in nums:
            if x % min_val != 0:
                # We can create a smaller positive value
                return 1
        
        # All values are divisible by min_val
        # Count how many times min_val appears
        count = nums.count(min_val)
        
        # We can pair up min_vals to get 0s
        # If count is odd, we'll have 1 element left
        # If count is even, we'll have 0 positive elements (but possibly some 0s)
        return (count + 1) // 2