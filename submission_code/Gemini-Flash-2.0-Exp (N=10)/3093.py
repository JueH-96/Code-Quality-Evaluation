class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for i, num in enumerate(nums):
            set_bits = 0
            temp = i
            while temp > 0:
                set_bits += temp & 1
                temp >>= 1
            if set_bits == k:
                total_sum += num
        return total_sum