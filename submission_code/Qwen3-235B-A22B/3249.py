class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_all = 0
        for num in nums:
            xor_all ^= num
        xor_diff = xor_all ^ k
        return bin(xor_diff).count('1')