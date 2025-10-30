class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        return int(''.join(map(lambda x: bin(x)[2:], sorted(nums, reverse=True))), 2)