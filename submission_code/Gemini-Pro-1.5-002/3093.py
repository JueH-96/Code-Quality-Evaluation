class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(len(nums)):
            set_bits = bin(i).count('1')
            if set_bits == k:
                ans += nums[i]
        return ans