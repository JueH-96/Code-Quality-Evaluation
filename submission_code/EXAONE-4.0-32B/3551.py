class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = nums[i]
        
        for length in range(2, n+1):
            for i in range(0, n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i][j-1] ^ dp[i+1][j]
        
        row_max = [[0] * n for _ in range(n)]
        for i in range(n):
            cur_max = dp[i][i]
            row_max[i][i] = cur_max
            for j in range(i+1, n):
                if dp[i][j] > cur_max:
                    cur_max = dp[i][j]
                row_max[i][j] = cur_max
        
        res_table = [[0] * n for _ in range(n)]
        for r in range(n):
            cur_max = -1
            for i in range(r, -1, -1):
                if row_max[i][r] > cur_max:
                    cur_max = row_max[i][r]
                res_table[i][r] = cur_max
        
        ans = []
        for l, r in queries:
            ans.append(res_table[l][r])
        return ans