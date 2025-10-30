class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        return sum(abs(num) for num in nums)