from typing import List

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        # Count frequencies of all values
        from collections import Counter
        freq = Counter(nums)

        # Separate out how many zeros we have (since 0 does not change sums)
        zero_count = freq.pop(0, 0)

        # Build a list of (value, count) for positive values only
        # (Order does not actually matter for correctness, but we'll sort for consistency)
        items = sorted(freq.items())

        # The maximum possible sum from positive elements
        S = sum(val * cnt for val, cnt in items)

        # dp[s] will hold the number of ways to form sum s using the processed distinct values so far
        dp = [0] * (S + 1)
        dp[0] = 1  # there's exactly 1 way to form sum = 0 (pick nothing)

        # For each distinct positive value x with frequency f,
        # update dp to account for picking 0..f copies of x.
        for x, f in items:
            new_dp = [0] * (S + 1)

            # We'll process dp in blocks of indices that share the same remainder mod x.
            # This is the standard "sliding window" trick for bounded knapsack-type counting.
            for start in range(x):
                # sumVal will track the sum of old_dp[s], old_dp[s-x], ... up to f+1 terms back
                sumVal = 0
                # Move s in steps of x, starting from 'start'
                for s in range(start, S + 1, x):
                    sumVal = (sumVal + dp[s]) % MOD
                    # If we've gone too far (more than f copies), subtract the oldest term
                    if s - x*(f+1) >= 0:
                        sumVal = (sumVal - dp[s - x*(f+1)]) % MOD
                    new_dp[s] = sumVal

            dp = new_dp

        # Now dp[s] is the count of ways (sub-multisets over positive nums) to get sum s.
        # Because zero_count zeros can be chosen in (zero_count+1) ways (0..zero_count copies)
        # for ANY sub-multiset, we multiply every dp[s] by (zero_count+1).
        # Finally, we sum the counts for sums in [l..r].
        ans = 0
        # Only sums up to S are possible from the positive elements
        low = max(l, 0)
        high = min(r, S)
        if low <= high:
            for s in range(low, high + 1):
                ans = (ans + dp[s]) % MOD

        # Multiply by the ways to pick zeros
        ans = (ans * (zero_count + 1)) % MOD
        return ans