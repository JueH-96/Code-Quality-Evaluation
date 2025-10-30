class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        bits = [0] * 32
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    bits[i] += 1
        
        result = 0
        mod = 10**9 + 7
        
        for _ in range(k):
            num = 0
            for i in range(32):
                if bits[i] > 0:
                    num += (1 << i)
                    bits[i] -= 1
            result = (result + num * num) % mod
        
        return result