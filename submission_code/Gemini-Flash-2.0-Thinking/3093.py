class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for i in range(len(nums)):
            # Count set bits in the index i
            set_bits_count = bin(i).count('1')
            if set_bits_count == k:
                total_sum += nums[i]
        return total_sum