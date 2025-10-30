class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        """
        We want to minimize the sum of nums by choosing, for each element, whether to:
          - Not apply any operation
          - Apply Operation 1 (divide by 2, rounding up) once
          - Apply Operation 2 (subtract k if >= k) once
          - Apply both operations (in whichever order yields the minimal final value) once

        We can apply Operation 1 at most 'op1' times (once per index) and Operation 2
        at most 'op2' times (once per index). The final sum should be as small as possible.

        Approach:
        1. For each element, determine the possible final values and how many uses of
           op1 and op2 they require (e.g., (0,0), (1,0), (0,1), (1,1)).
        2. Use DP over the array indices and the number of used op1/op2 so far:
           dp[i][j][m] = minimum sum achievable using the first i elements with j uses
           of Operation 1 and m uses of Operation 2.
        3. Transition by trying each valid choice for element i.
        4. Answer is min(dp[n][j][m]) for 0 <= j <= op1, 0 <= m <= op2.

        Complexity is O(n * op1 * op2 * 4) which is acceptable for n <= 100 and op1, op2 <= n.
        """

        import math

        n = len(nums)

        # Precompute possible (cost, (use_op1, use_op2)) for each index
        # We'll consider up to four scenarios for each element:
        #   (a) No operation
        #   (b) Operation 1 (divide by 2, round up)
        #   (c) Operation 2 (subtract k, if >= k)
        #   (d) Both ops (in whichever order yields the smaller result), if valid
        cost_options = []
        for x in nums:
            options = []
            # (a) No operation
            options.append((x, (0, 0)))

            # (b) Operation 1
            div_val = (x + 1) // 2
            if div_val < x:
                options.append((div_val, (1, 0)))

            # (c) Operation 2 (only if x >= k)
            if x >= k:
                options.append((x - k, (0, 1)))

            # (d) Both operations
            #    We'll consider two orders:
            #    1) op1 -> op2 if ceil(x/2) >= k
            #    2) op2 -> op1 if x >= k
            #    We take the min final value if both are valid
            c_vals = []
            # op1 -> op2
            after_div = (x + 1) // 2
            if after_div >= k:
                c_vals.append(after_div - k)
            # op2 -> op1
            if x >= k:
                after_sub = x - k
                c_vals.append((after_sub + 1) // 2)
            if c_vals:
                best_both = min(c_vals)
                options.append((best_both, (1, 1)))

            cost_options.append(options)

        # Large number for initialization
        INF = 10**15
        # dp[i][j][m] = minimal sum for first i elements with j times using op1, m times using op2
        dp = [[[INF] * (op2 + 1) for _ in range(op1 + 1)] for __ in range(n + 1)]
        dp[0][0][0] = 0

        # Build up the DP
        for i in range(n):
            for used1 in range(op1 + 1):
                for used2 in range(op2 + 1):
                    if dp[i][used1][used2] == INF:
                        continue
                    curr_sum = dp[i][used1][used2]
                    for cost_val, (u1, u2) in cost_options[i]:
                        new1 = used1 + u1
                        new2 = used2 + u2
                        if new1 <= op1 and new2 <= op2:
                            dp[i+1][new1][new2] = min(dp[i+1][new1][new2],
                                                     curr_sum + cost_val)

        # The answer is the minimal dp[n][j][m] for all j <= op1, m <= op2
        ans = min(dp[n][j][m] for j in range(op1 + 1) for m in range(op2 + 1))
        return ans