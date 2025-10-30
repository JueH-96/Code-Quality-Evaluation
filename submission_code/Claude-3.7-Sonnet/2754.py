class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # If there's only one number, return it
        if n == 1:
            return nums[0]
        
        positives = [num for num in nums if num > 0]
        negatives = [num for num in nums if num < 0]
        zero_count = nums.count(0)
        
        # If there are no positive numbers and no negative numbers (all are zeros), return 0
        if not positives and not negatives:
            return 0
        
        # If there are no positive numbers and only one negative number
        if not positives and len(negatives) == 1:
            if zero_count > 0:
                return 0  # Choosing a zero gives 0, which is better than the negative number
            else:
                return negatives[0]  # No choice but to choose the negative number
        
        # Start with the product of all positive numbers
        product = 1
        for num in positives:
            product *= num
        
        # Sort negatives by absolute value in descending order
        negatives.sort(key=lambda x: abs(x), reverse=True)
        
        # If there's an odd number of negative numbers, exclude the one with the smallest absolute value
        if len(negatives) % 2 == 1:
            negatives.pop(-1)
        
        # Multiply by the remaining negative numbers
        for num in negatives:
            product *= num
        
        return product