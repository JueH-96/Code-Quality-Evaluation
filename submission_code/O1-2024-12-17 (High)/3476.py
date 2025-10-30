class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total_operations = 0
        for num in nums:
            remainder = num % 3
            if remainder != 0:
                total_operations += min(remainder, 3 - remainder)
        return total_operations