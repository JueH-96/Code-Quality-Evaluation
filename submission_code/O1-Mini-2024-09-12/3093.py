class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        return sum(num for idx, num in enumerate(nums) if bin(idx).count('1') == k)