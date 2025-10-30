class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        """
        We want to find the maximum sum x we can achieve by picking rewards in some order,
        starting from x=0, with the condition that we can only pick rewardValues[i] if it is
        strictly greater than our current total x.

        Key Insight:
        If we ever choose rewards r1, r2, ..., rK in that order, each must satisfy:
            r1 > 0,
            r2 > r1,
            r3 > r1 + r2,
            ...
            rK > r1 + r2 + ... + r_{K-1}.
        In particular, r1 < r2 < r3 < ... < rK (strictly increasing in the order they are used).
        Therefore, we lose no generality by sorting the array first and only considering
        a left-to-right "pick or skip" in ascending order.

        However, simply "greedily picking" every time a reward is greater than the current sum
        may fail to maximize the final total (see the second example in the prompt).

        A correct approach:
        1) Sort the rewards in ascending order.
        2) Use dynamic programming over (index, currentSumSoFar).
        3) Because the largest reward is at most 2000, once currentSumSoFar >= 4001, no further
           picks are possible. (Since no reward can exceed 2000, we cannot pick anything
           larger than that new sum.)
        4) We work bottom-up. Let dp[i][s] = the maximum final sum achievable starting at index i
           with current sum = s, using rewards[i..n-1].
           Then:
               dp[i][s] = max(
                   dp[i+1][s],          # skip reward i
                   if reward[i] > s:
                       if s + reward[i] <= 4000:
                           dp[i+1][s + reward[i]]
                       else:
                           (s + reward[i])  # we exceed 4000, so we won't pick any more
               )
           Base case: dp[n][s] = s  (no more items to pick).
        5) The answer is dp[0][0].

        Complexity:
        - n <= 2000
        - We track sums s up to 4000
        - Thus we have O(n * 4001) = around 8 million states. Each state does O(1) work,
          which can be borderline but typically can be done efficiently enough in a compiled
          language. In Python, careful implementation can still pass moderate time limits.

        This DP correctly accounts for possibly skipping smaller items to allow picking
        bigger ones later, hence maximizing the final sum.

        Examples:
          Example 1: [1,1,3,3] -> answer 4
          Example 2: [1,6,4,3,2] -> answer 11
        """

        # Sort the rewards in ascending order
        rewards = sorted(rewardValues)
        n = len(rewards)

        # We only need to track dp for sums up to 4000.
        # dp[i][s] = the maximum final sum achievable using items from i..n-1,
        # starting with current sum = s.
        # We'll do a rolling array of size 4001 to save memory.
        MAX_SUM = 4000

        # dp_next[s] will correspond to dp[i+1][s]
        # dp_curr[s] will correspond to dp[i][s]
        dp_next = [0] * (MAX_SUM + 1)

        # Base case: i = n => dp[n][s] = s
        for s in range(MAX_SUM + 1):
            dp_next[s] = s

        # Fill from the back: i = n-1 down to 0
        for i in reversed(range(n)):
            val = rewards[i]
            dp_curr = [0] * (MAX_SUM + 1)
            for s in range(MAX_SUM + 1):
                # Option 1: skip
                best = dp_next[s]

                # Option 2: pick if val > s
                if val > s:
                    new_sum = s + val
                    if new_sum <= MAX_SUM:
                        # we can continue picking afterwards
                        best = max(best, dp_next[new_sum])
                    else:
                        # once we exceed MAX_SUM, we won't pick anything else
                        best = max(best, new_sum)

                dp_curr[s] = best
            dp_next = dp_curr

        # The result is dp[0][0]
        return dp_next[0]