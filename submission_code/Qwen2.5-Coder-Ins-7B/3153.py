class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        freq = [0] * 32
        
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    freq[i] += 1
        
        result = 0
        for _ in range(k):
            value = 0
            for i in range(32):
                if freq[i] > 0:
                    value |= (1 << i)
                    freq[i] -= 1
            result = (result + value * value) % MOD
        
        return result