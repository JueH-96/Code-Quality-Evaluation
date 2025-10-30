class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total_ops = 0
        for num in nums:
            remainder = num % 3
            if remainder == 1:
                total_ops += 1
            elif remainder == 2:
                total_ops += 1
        return total_ops