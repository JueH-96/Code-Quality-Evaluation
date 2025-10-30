class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = 0
        for num in nums:
            remainder = num % 3
            if remainder != 0:
                count += min(remainder, 3 - remainder)
        return count