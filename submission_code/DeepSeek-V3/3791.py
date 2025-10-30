class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1: Replace even numbers with 0 and odd numbers with 1
        transformed = [0 if num % 2 == 0 else 1 for num in nums]
        # Step 2: Sort the transformed array in non-decreasing order
        transformed.sort()
        return transformed