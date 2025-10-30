class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We want to find the smallest k >= 1 such that:
        #   M = num1 - k*num2 >= 0
        #   M >= k
        #   popcount(M) <= k
        #
        # Reasoning:
        # After k operations, each subtracting (2^i + num2),
        # we require:  num1 - sum(2^i_j + num2) = 0
        # which rearranges to:  sum(2^i_j) + k*num2 = num1
        # so sum(2^i_j) = M = num1 - k*num2.
        #
        # Each 2^i_j is at least 1 (since i>=0 => 2^i >=1),
        # so if we use k terms, their sum is at least k.
        # Hence M >= k.  Also, to form M from powers of two,
        # we need at least popcount(M) terms, so we require
        # popcount(M) <= k.  Finally, M must be nonnegative.
        #
        # A key observation (common in contest/editorial solutions)
        # is that checking k up to around 60 or so is sufficient
        # under these constraints.  (Because num1 ≤ 1e9 and
        # num2 ≥ -1e9, larger k will not yield a smaller solution
        # once these bit constraints are considered.)
        #
        # Implementation details:
        # 1) Loop k in [1..60].
        # 2) Compute M = num1 - k*num2.
        # 3) If M < 0 and num2 > 0, break (M will only get smaller).
        #    If M < 0 and num2 <= 0, keep going (M may increase for a bigger k).
        # 4) If M < k, continue (sum of k powers of two can’t be < k).
        # 5) If popcount(M) <= k, return k.
        # If no k works, return -1.

        import math

        # A quick helper for popcount; Python 3.10+ has int.bit_count()
        # For older versions, one could do: bin(M).count('1')
        for k in range(1, 61):
            M = num1 - k*num2
            if M < 0:
                # If num2 > 0 and M is already negative, increasing k
                # only decreases M further, so break out.
                if num2 > 0:
                    break
                # If num2 <= 0, M might become larger for bigger k,
                # so continue checking.
                continue

            if M < k:
                continue

            # Check if the number of set bits in M is <= k
            if M.bit_count() <= k:
                return k

        return -1