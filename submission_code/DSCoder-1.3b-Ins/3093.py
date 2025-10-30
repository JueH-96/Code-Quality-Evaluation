class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        return sum(num for index, num in enumerate(nums) if bin(index).count('1') == k)