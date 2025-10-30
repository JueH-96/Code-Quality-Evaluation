class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        result = 0
        for i, num in enumerate(nums):
            # Count set bits in the binary representation of index i
            if bin(i).count('1') == k:
                result += num
        return result