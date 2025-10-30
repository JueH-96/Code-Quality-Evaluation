class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        bits = [0] * 32
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    bits[i] += 1
        ans = 0
        for _ in range(k):
            curr = 0
            for i in range(32):
                if bits[i] > 0:
                    curr |= (1 << i)
                    bits[i] -= 1
            ans = (ans + curr * curr) % MOD
        return ans