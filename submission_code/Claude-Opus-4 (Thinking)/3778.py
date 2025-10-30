class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace even with 0, odd with 1
        transformed = [num % 2 for num in nums]
        
        # Sort the array in non-decreasing order
        transformed.sort()
        
        return transformed