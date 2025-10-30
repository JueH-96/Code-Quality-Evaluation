class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        """
        We use dynamic programming to decide, for each element, which operation(s) to apply:
        - Do nothing
        - Apply operation 1 (halve, rounding up)
        - Apply operation 2 (subtract k if >= k)
        - Apply both operations in the order that yields the smallest final value (if valid)

        Let dp[i][c1][c2] = minimum sum of the first i elements if we have used c1 of the 'halve' ops
        and c2 of the 'subtract k' ops so far. We will compute all valid ways of assigning operations
        to nums[i] and update dp accordingly. In the end, we take the minimum dp[n][c1][c2] for all
        c1 <= op1 and c2 <= op2.
        """
        import math

        n = len(nums)
        INF = 10**15

        # dp[i][x][y] = minimal sum for first i elements using x operation1's and y operation2's
        dp = [[[INF] * (op2 + 1) for _ in range(op1 + 1)] for _ in range(n + 1)]
        dp[0][0][0] = 0

        def half_ceil(x):
            return (x + 1) // 2

        for i in range(n):
            v = nums[i]

            # Scenario 0: Do nothing
            cost0 = v
            # Scenario 1: Halve
            cost1 = half_ceil(v)
            # Scenario 2: Subtract k (valid only if v >= k)
            cost2 = v - k if v >= k else None
            # Scenario 3: Apply both ops, in which order results in the smallest outcome?
            #   op1 -> op2: finalA = half_ceil(v) - k if half_ceil(v) >= k
            #   op2 -> op1: finalB = half_ceil(v - k) if v >= k
            candidates_3 = []
            if half_ceil(v) >= k:
                candidates_3.append(half_ceil(v) - k)
            if v >= k:
                candidates_3.append(half_ceil(v - k))
            cost3 = min(candidates_3) if candidates_3 else None

            for used1 in range(op1 + 1):
                for used2 in range(op2 + 1):
                    if dp[i][used1][used2] == INF:
                        continue

                    base_val = dp[i][used1][used2]

                    # Scenario 0
                    dp[i+1][used1][used2] = min(dp[i+1][used1][used2], base_val + cost0)

                    # Scenario 1 (halve)
                    if used1 < op1:
                        dp[i+1][used1+1][used2] = min(
                            dp[i+1][used1+1][used2],
                            base_val + cost1
                        )

                    # Scenario 2 (subtract k)
                    if cost2 is not None and used2 < op2:
                        dp[i+1][used1][used2+1] = min(
                            dp[i+1][used1][used2+1],
                            base_val + cost2
                        )

                    # Scenario 3 (both)
                    if cost3 is not None and used1 < op1 and used2 < op2:
                        dp[i+1][used1+1][used2+1] = min(
                            dp[i+1][used1+1][used2+1],
                            base_val + cost3
                        )

        # We want the minimal sum among all ways of using at most op1 times operation 1 and at most op2 times operation 2
        ans = INF
        for c1 in range(op1 + 1):
            for c2 in range(op2 + 1):
                ans = min(ans, dp[n][c1][c2])

        return ans