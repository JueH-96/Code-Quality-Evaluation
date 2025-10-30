class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i in range(n) if nums[i] == 1]
        if len(ones) < k:
            return -1
        
        def cost(i, j):
            return abs(i - j)
        
        def min_cost_subarray(arr, k):
            n = len(arr)
            dp = [[float('inf')] * (k + 1) for _ in range(n)]
            dp[0][1] = cost(0, arr[0])
            for i in range(1, n):
                dp[i][1] = min(dp[i - 1][1], cost(i, arr[i]))
                for j in range(2, k + 1):
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1] + cost(i, arr[i]))
            return dp[n - 1][k]
        
        def max_changes_cost(arr, maxChanges):
            n = len(arr)
            dp = [[float('inf')] * (maxChanges + 1) for _ in range(n)]
            dp[0][1] = cost(0, arr[0])
            for i in range(1, n):
                dp[i][1] = min(dp[i - 1][1], cost(i, arr[i]))
                for j in range(2, maxChanges + 1):
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1] + cost(i, arr[i]))
            return dp[n - 1][maxChanges]
        
        ones_cost = min_cost_subarray(ones, k)
        changes_cost = max_changes_cost(ones, maxChanges)
        return ones_cost + changes_cost