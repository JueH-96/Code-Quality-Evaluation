class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        transformed = [num % 2 for num in nums]
        transformed.sort()
        return transformed