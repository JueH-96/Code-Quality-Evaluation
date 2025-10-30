class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace even numbers with 0 and odd numbers with 1
        transformed = [0 if num % 2 == 0 else 1 for num in nums]
        # Sort the transformed array in non-decreasing order
        transformed.sort()
        return transformed