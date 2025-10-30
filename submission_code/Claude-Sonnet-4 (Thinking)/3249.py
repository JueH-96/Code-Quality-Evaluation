class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        diff = current_xor ^ k
        
        return bin(diff).count('1')