class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        sum_val = 0
        for i in range(len(nums)):
            binary = bin(i)[2:]
            set_bits = 0
            for bit in binary:
                if bit == '1':
                    set_bits += 1
            if set_bits == k:
                sum_val += nums[i]
        return sum_val