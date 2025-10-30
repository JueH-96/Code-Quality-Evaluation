class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        # Separate positive and negative numbers
        positives = [num for num in nums if num > 0]
        negatives = [num for num in nums if num < 0]
        
        # If there are no positive numbers, we need to consider the largest negative number
        if not positives:
            return max(negatives)
        
        # Sort negatives to maximize the product
        negatives.sort()
        
        # If the count of negatives is odd, we need to exclude the largest (least negative)
        if len(negatives) % 2 != 0:
            negatives.pop()
        
        # Calculate the product of positives and the selected negatives
        product = 1
        for num in positives + negatives:
            product *= num
        
        return product