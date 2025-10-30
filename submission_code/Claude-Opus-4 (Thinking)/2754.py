class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        positives = [x for x in nums if x > 0]
        negatives = [x for x in nums if x < 0]
        has_zero = 0 in nums
        
        # If only zeros
        if not positives and not negatives:
            return 0
        
        # Start with all positive numbers
        result = 1
        for p in positives:
            result *= p
        
        # Sort negatives (most negative first)
        negatives.sort()
        
        # Special case: no positives and odd number of negatives
        if not positives and len(negatives) % 2 == 1:
            if len(negatives) == 1:
                # Only one negative number
                return 0 if has_zero else negatives[0]
            
            # Multiple negatives - we have two options:
            # Option 1: Take all negatives (negative product)
            option1 = 1
            for n in negatives:
                option1 *= n
            
            # Option 2: Exclude the least negative one (positive product)
            option2 = 1
            for i in range(len(negatives) - 1):
                option2 *= negatives[i]
            
            # If we have zero, that's another option
            if has_zero:
                return max(0, option1, option2)
            return max(option1, option2)
        else:
            # We have positives or even number of negatives
            # Include negatives in pairs (excluding the least negative if odd count)
            for i in range(0, len(negatives) - len(negatives) % 2, 2):
                result *= negatives[i] * negatives[i + 1]
            
            return result