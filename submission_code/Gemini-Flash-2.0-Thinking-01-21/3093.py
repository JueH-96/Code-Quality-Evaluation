class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for index, num in enumerate(nums):
            binary_index = bin(index)[2:]
            set_bits_count = binary_index.count('1')
            if set_bits_count == k:
                total_sum += num
        return total_sum