class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        sum_val = 0
        for i in range(len(nums)):
            set_bits = 0
            index = i
            while index > 0:
                if index & 1:
                    set_bits += 1
                index >>= 1
            if set_bits == k:
                sum_val += nums[i]
        return sum_val