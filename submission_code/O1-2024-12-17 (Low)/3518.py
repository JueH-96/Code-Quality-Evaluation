class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # We want to pick indices i0 < i1 < i2 < i3 from b to maximize:
        # a[0]*b[i0] + a[1]*b[i1] + a[2]*b[i2] + a[3]*b[i3].
        #
        # We can use a DP approach with 4 stages:
        # Let dp1 track the best value using one term (a[0]),
        #     dp2 track the best value using two terms (a[0], a[1]),
        #     dp3 track the best value using three terms (a[0], a[1], a[2]),
        #     dp4 track the best value using four terms (a[0], a[1], a[2], a[3]).
        #
        # As we iterate through b, we update:
        #   dp1 = max(dp1, 0 + a[0]*b[i])
        #   dp2 = max(dp2, dp1_old + a[1]*b[i])
        #   dp3 = max(dp3, dp2_old + a[2]*b[i])
        #   dp4 = max(dp4, dp3_old + a[3]*b[i])
        #
        # The final answer will be dp4 after we process the entire b array.

        dp1 = float('-inf')
        dp2 = float('-inf')
        dp3 = float('-inf')
        dp4 = float('-inf')

        for val in b:
            # We'll make updates in descending order or use temp variables
            # so as not to overwrite values we still need.
            dp4_new = max(dp4, dp3 + a[3]*val if dp3 != float('-inf') else float('-inf'))
            dp3_new = max(dp3, dp2 + a[2]*val if dp2 != float('-inf') else float('-inf'))
            dp2_new = max(dp2, dp1 + a[1]*val if dp1 != float('-inf') else float('-inf'))
            dp1_new = max(dp1, a[0]*val)  # dp0 is effectively 0, so "dp0+a[0]*val" is just a[0]*val

            dp1, dp2, dp3, dp4 = dp1_new, dp2_new, dp3_new, dp4_new

        return dp4