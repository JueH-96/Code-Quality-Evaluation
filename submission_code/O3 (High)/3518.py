from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        """
        Dynamic programming with a constant sized state (4 elements to pick).
        dp[k] – best value after having already chosen (k + 1) elements
                whose indices are strictly increasing and are not past
                the current position in b that we are processing.
        We process b left-to-right and update dp backwards (k = 3 → 0) so
        every element of b is considered at most once for each stage.
        """
        NEG = -10 ** 18               # sufficiently small sentinel
        dp = [NEG, NEG, NEG, NEG]     # dp[0]..dp[3] for 1..4 chosen elements

        for x in b:
            # update in reverse order to avoid re-using the same element twice
            if dp[2] != NEG:                 # we have already chosen 3 elements
                dp[3] = max(dp[3], dp[2] + a[3] * x)
            if dp[1] != NEG:                 # we have already chosen 2 elements
                dp[2] = max(dp[2], dp[1] + a[2] * x)
            if dp[0] != NEG:                 # we have already chosen 1 element
                dp[1] = max(dp[1], dp[0] + a[1] * x)
            # choose current position as the 1st element
            dp[0] = max(dp[0], a[0] * x)

        return dp[3]     # maximum score using exactly 4 positions