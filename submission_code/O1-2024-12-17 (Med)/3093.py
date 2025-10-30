class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for i in range(len(nums)):
            # Count the number of set bits in the binary representation of i
            if bin(i).count('1') == k:
                total_sum += nums[i]
        return total_sum