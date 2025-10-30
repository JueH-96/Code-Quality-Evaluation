class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        nums.sort()
        dp = [[0]*64 for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            for j in range(63, -1, -1):
                dp[i][j] = max(dp[i+1][j], ((nums[i]>>j)&1) * nums[i] + dp[i+1][j+1])
        res = 0
        for i in range(63, -1, -1):
            cnt = sum(((num>>i)&1) for num in nums)
            if cnt >= k:
                res += (1<<i)*cnt
                k = 0
            else:
                res += (1<<i)*cnt
                k -= cnt
                j = 0
                for num in nums:
                    if ((num>>(i+1))&1) == 0 and j < cnt:
                        res += (1<<i) * ((num>>i)&1)
                        j += 1
        return res % mod