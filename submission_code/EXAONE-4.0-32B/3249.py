class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_val = 0
        for num in nums:
            xor_val ^= num
        diff = xor_val ^ k
        return diff.bit_count()