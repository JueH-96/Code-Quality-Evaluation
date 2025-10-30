class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total_ops = 0
        for num in nums:
            r = num % 3
            total_ops += min(r, 3 - r)
        return total_ops