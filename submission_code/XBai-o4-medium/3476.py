class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            r = num % 3
            total += min(r, 3 - r)
        return total