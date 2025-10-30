class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_val = max(nums)
        spf = list(range(max_val + 1))
        if max_val >= 2:
            for i in range(2, int(max_val**0.5) + 1):
                if spf[i] == i:
                    for j in range(i*i, max_val + 1, i):
                        if spf[j] == j:
                            spf[j] = i
        
        INF = 10**9
        dp = [[INF] * 2 for _ in range(n)]
        
        if nums[0] == 1 or spf[nums[0]] == nums[0]:
            dp[0][0] = 0
            dp[0][1] = INF
        else:
            dp[0][0] = 0
            dp[0][1] = 1
        
        for i in range(1, n):
            min_ops0 = INF
            if nums[i] >= nums[i-1]:
                min_ops0 = min(min_ops0, dp[i-1][0])
            if dp[i-1][1] != INF:
                prev_val = spf[nums[i-1]]
                if nums[i] >= prev_val:
                    min_ops0 = min(min_ops0, dp[i-1][1])
            dp[i][0] = min_ops0

            if nums[i] != 1 and spf[nums[i]] < nums[i]:
                min_ops1 = INF
                current_val = spf[nums[i]]
                if current_val >= nums[i-1]:
                    min_ops1 = min(min_ops1, dp[i-1][0] + 1)
                if dp[i-1][1] != INF:
                    prev_val = spf[nums[i-1]]
                    if current_val >= prev_val:
                        min_ops1 = min(min_ops1, dp[i-1][1] + 1)
                dp[i][1] = min_ops1
            else:
                dp[i][1] = INF
        
        ans = min(dp[n-1][0], dp[n-1][1])
        return ans if ans != INF else -1