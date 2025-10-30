class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        T = [0] * 31
        
        # Calculate the count of set bits for each position
        for num in nums:
            for bit in range(31):
                if num & (1 << bit):
                    T[bit] += 1
        
        res = [0] * k
        
        # Process each bit from highest to lowest
        for bit in reversed(range(31)):
            val = 1 << bit
            cnt = T[bit]
            for i in range(k):
                if cnt <= 0:
                    break
                res[i] += val
                cnt -= 1
        
        # Calculate the sum of squares modulo MOD
        total = 0
        for x in res:
            total = (total + x * x) % MOD
        
        return total