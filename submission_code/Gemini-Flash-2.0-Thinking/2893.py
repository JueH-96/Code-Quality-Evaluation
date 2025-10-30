class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [[float('-inf')] * 2 for _ in range(n)]

        dp[0][nums[0] % 2] = nums[0]

        for i in range(1, n):
            current_parity = nums[i] % 2
            opposite_parity = 1 - current_parity

            max_prev_current = float('-inf')
            for j in range(i):
                if dp[j][current_parity] != float('-inf'):
                    max_prev_current = max(max_prev_current, dp[j][current_parity])

            max_prev_opposite = float('-inf')
            for j in range(i):
                if dp[j][opposite_parity] != float('-inf'):
                    max_prev_opposite = max(max_prev_opposite, dp[j][opposite_parity])

            if max_prev_current != float('-inf'):
                dp[i][current_parity] = max(dp[i][current_parity], nums[i] + max_prev_current)

            if max_prev_opposite != float('-inf'):
                dp[i][current_parity] = max(dp[i][current_parity], nums[i] - x + max_prev_opposite)

        return max(dp[n - 1])