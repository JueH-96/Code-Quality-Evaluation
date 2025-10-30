class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        xor_diff = current_xor ^ k
        return bin(xor_diff).count('1')