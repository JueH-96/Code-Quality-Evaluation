class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bit_counts = [0] * 31
        for num in nums:
            for b in range(31):
                if num & (1 << b):
                    bit_counts[b] += 1
        
        result = [0] * k
        for b in range(30, -1, -1):
            c = bit_counts[b]
            m = min(c, k)
            for i in range(m):
                result[i] += (1 << b)
        
        total = 0
        for num in result:
            total = (total + num * num) % MOD
        return total