class Solution:
    def minimumOperations(self, num: str) -> int:
        """
        We want to transform num into a string that represents an integer divisible by 25,
        by deleting digits (possibly none). The special integer must end with "00", "25",
        "50", or "75", or it can be just "0" (since 0 is divisible by 25), and deleting
        all digits yields 0 as well.

        Approach:
          1) We check the costs to form each of the patterns "00", "25", "50", "75" from
             the end of the string. For each pattern, we:
               - Find the rightmost digit of the pattern by scanning from the end of num.
               - Once found, find the left digit (of the pattern) by continuing scanning
                 left.
               - The total number of skipped digits along the way is the cost for that pattern.
               - If we cannot form it, we mark the cost as large (invalid).
          2) We also consider that we can either:
               - Delete all digits (cost = len(num)) → resulting in "0" which is divisible by 25.
               - Keep exactly one '0' if it exists (cost = len(num)-1) → "0".
          3) We take the minimum of all these possible costs.

        Example:
          num = "2245047"
            - minimumOperations -> 2
            Explanation: "2245047" can become "22450" (divisible by 25) by deleting 2 digits.
        """

        patterns = ["00", "25", "50", "75"]
        n = len(num)

        def cost_to_form(pattern: str) -> int:
            # We try to match pattern[1] (the rightmost digit) first, then pattern[0].
            # Skip digits in between to see if we can form that ending.
            steps = 0
            j = 1  # We'll match the second char of pattern first
            i = n - 1
            # match pattern[1]
            while i >= 0 and num[i] != pattern[j]:
                i -= 1
                steps += 1
            if i < 0:  # not found
                return float("inf")

            # Now match pattern[0]
            i -= 1
            while i >= 0 and num[i] != pattern[0]:
                i -= 1
                steps += 1
            if i < 0:  # not found
                return float("inf")

            return steps

        # Check costs for each of the four patterns
        ans = min(cost_to_form(p) for p in patterns)

        # Cost if we skip all digits (forming "0")
        all_deleted_cost = n

        # Cost if we keep just one zero (if any '0' is present)
        if '0' in num:
            keep_one_zero_cost = n - 1
            ans = min(ans, keep_one_zero_cost, all_deleted_cost)
        else:
            ans = min(ans, all_deleted_cost)

        return ans