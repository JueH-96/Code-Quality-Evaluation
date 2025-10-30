class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_sum = 0
        for num in nums:
            xor_sum ^= num
        
        diff = xor_sum ^ k
        
        count = 0
        while diff > 0:
            if diff & 1:
                count += 1
            diff >>= 1
            
        return count