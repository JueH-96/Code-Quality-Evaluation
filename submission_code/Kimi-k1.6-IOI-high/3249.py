class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_sum = 0
        for num in nums:
            xor_sum ^= num
        diff = xor_sum ^ k
        return bin(diff).count('1')