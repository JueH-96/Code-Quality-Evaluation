class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for i, val in enumerate(nums):
            if bin(i).count('1') == k:
                total_sum += val
        return total_sum