class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for index, value in enumerate(nums):
            binary_representation = bin(index)
            set_bits = binary_representation.count('1')
            if set_bits == k:
                total_sum += value
        return total_sum