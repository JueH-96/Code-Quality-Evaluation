class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7
        ans = 0
        cnt = Counter()
        for num in nums:
            for bit in range(30):
                if num & (1 << bit):
                    cnt[bit] += 1
        for _ in range(k):
            cur = 0
            for bit in range(30):
                if cnt[bit] > 0:
                    cur += (1 << bit)
                    cnt[bit] -= 1
            ans = (ans + pow(cur, 2, mod)) % mod
        return ans