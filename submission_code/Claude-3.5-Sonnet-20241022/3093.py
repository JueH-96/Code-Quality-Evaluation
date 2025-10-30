class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        result = 0
        for i in range(len(nums)):
            if bin(i).count('1') == k:
                result += nums[i]
        return result