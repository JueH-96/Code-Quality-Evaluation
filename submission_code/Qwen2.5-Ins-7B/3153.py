class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        bit_counts = [0] * 30
        for num in nums:
            for i in range(30):
                if num & (1 << i):
                    bit_counts[i] += 1
        
        result = 0
        mod = 10**9 + 7
        for _ in range(k):
            max_val = 0
            for i in range(30):
                if bit_counts[i]:
                    max_val |= (1 << i)
                    bit_counts[i] -= 1
            result = (result + max_val * max_val) % mod
        
        return result