class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_sum = 0
        for num in nums:
            xor_sum ^= num
        
        xor_result = xor_sum ^ k
        
        count = 0
        while xor_result > 0:
            count += xor_result & 1
            xor_result >>= 1
        
        return count