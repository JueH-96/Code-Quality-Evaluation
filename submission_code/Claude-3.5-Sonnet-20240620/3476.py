class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        for num in nums:
            remainder = num % 3
            if remainder == 1:
                operations += min(1, 2)  # Add 1 or subtract 2
            elif remainder == 2:
                operations += min(1, 1)  # Add 1 or subtract 1
        return operations