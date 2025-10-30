class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp1 = dp2 = dp3 = dp4 = float('-inf')
        for num in b:
            new_dp1 = max(dp1, a[0] * num)
            new_dp2 = max(dp2, dp1 + a[1] * num)
            new_dp3 = max(dp3, dp2 + a[2] * num)
            new_dp4 = max(dp4, dp3 + a[3] * num)
            dp1, dp2, dp3, dp4 = new_dp1, new_dp2, new_dp3, new_dp4
        return dp4