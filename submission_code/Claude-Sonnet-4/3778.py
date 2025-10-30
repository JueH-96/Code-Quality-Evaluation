class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace even numbers with 0 and odd numbers with 1
        transformed = []
        for num in nums:
            if num % 2 == 0:  # even
                transformed.append(0)
            else:  # odd
                transformed.append(1)
        
        # Sort in non-decreasing order
        transformed.sort()
        
        return transformed