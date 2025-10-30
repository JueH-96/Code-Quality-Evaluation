class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_result = 0
        for num in nums:
            xor_result ^= num
        xor_result ^= k
        return xor_result.bit_count()