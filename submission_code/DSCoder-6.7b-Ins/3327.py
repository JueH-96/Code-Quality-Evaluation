class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        q = collections.deque([0])
        for i in range(1, n + 1):
            while q and i - q[0] > k:
                q.popleft()
            dp[i] = dp[q[0]] + 1
            while q and dp[q[-1]] >= dp[i] - (prefix[i] - prefix[q[-1]]):
                q.pop()
            q.append(i)
        return dp[-1]