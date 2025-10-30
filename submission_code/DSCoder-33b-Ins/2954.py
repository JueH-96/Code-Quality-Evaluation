class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        dp = [0] * (n + 1)
        dp2 = [0] * (n + 1)
        q = collections.deque()
        for i in range(1, n + 1):
            while q and i - q[0] >= k:
                q.popleft()
            if i >= k:
                dp[i] = max(dp[i - 1], prefix_sum[i] - prefix_sum[i - k] + dp2[i - k])
            if q:
                dp[i] = max(dp[i], dp2[q[0]])
            while q and dp2[i] >= dp2[q[-1]]:
                q.pop()
            q.append(i)
            dp2[i] = dp[i]

        return dp[-1]