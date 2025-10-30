class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bit_counts = [0] * 32
        
        # Count the number of 1s in each bit position
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    bit_counts[i] += 1
        
        result = 0
        for _ in range(k):
            current = 0
            for i in range(32):
                if bit_counts[i] > 0:
                    current |= (1 << i)
                    bit_counts[i] -= 1
            result = (result + current * current) % MOD
        
        return result