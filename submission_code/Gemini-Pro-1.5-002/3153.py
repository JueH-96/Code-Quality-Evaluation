class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        bits = [0] * 32
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    bits[i] += 1
        
        res = [0] * n
        for i in range(k):
            curr = 0
            for j in range(32):
                if bits[j] > 0:
                    curr |= (1 << j)
                    bits[j] -= 1
            res[i] = curr
        
        ans = 0
        mod = 10**9 + 7
        for num in res:
            ans = (ans + num * num) % mod
        
        return ans