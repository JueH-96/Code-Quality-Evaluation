class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        import math
        
        # Find GCD of all elements
        gcd_all = nums[0]
        for num in nums[1:]:
            gcd_all = math.gcd(gcd_all, num)
        
        # Find minimum element
        min_val = min(nums)
        
        # If GCD < minimum, we can reduce to single element
        if gcd_all < min_val:
            return 1
        
        # If GCD = minimum, count occurrences of minimum
        # and apply the formula
        count_min = nums.count(min_val)
        return (count_min + 1) // 2