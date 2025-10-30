class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_sum = 0
        for num in nums:
            xor_sum ^= num
        
        diff = xor_sum ^ k
        
        count = 0
        
        if diff == 0:
            return 0

        
        bits = 0
        temp = diff
        while temp > 0:
            bits += temp & 1
            temp >>= 1
        return bits