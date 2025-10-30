class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        for num in nums:
            remainder = num % 3
            if remainder == 0:
                continue
            operations += min(remainder, 3 - remainder)
        return operations