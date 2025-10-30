class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        target_xor = current_xor ^ k
        return bin(target_xor).count('1')