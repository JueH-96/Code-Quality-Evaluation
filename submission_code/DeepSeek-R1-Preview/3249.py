class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        target_diff = current_xor ^ k
        return bin(target_diff).count('1')