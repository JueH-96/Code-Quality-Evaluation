class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total_operations = 0
        for num in nums:
            if num % 3 != 0:
                total_operations += 1
        return total_operations