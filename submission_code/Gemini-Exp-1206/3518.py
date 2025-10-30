class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp = {}

        def solve(idx, b_idx):
            if idx == 4:
                return 0
            if (idx, b_idx) in dp:
                return dp[(idx, b_idx)]

            ans = float('-inf')
            for i in range(b_idx, n - (3 - idx)):
                ans = max(ans, a[idx] * b[i] + solve(idx + 1, i + 1))

            dp[(idx, b_idx)] = ans
            return ans

        return solve(0, 0)