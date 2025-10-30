from typing import List, Set, Tuple

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        # dp[j] will be a set of tuples (or1, or2)
        # dp[j]: possible OR sums for subsequences of size j using elements considered so far.
        # or1: OR sum of the first min(j, k) elements.
        # or2: OR sum of the last max(0, j-k) elements.

        # Initialize DP table: a list of sets, size 2k + 1
        dp: List[Set[Tuple[int, int]]] = [set() for _ in range(2 * k + 1)]

        # Base case: subsequence of size 0 has OR sums (0, 0)
        # The OR sum of an empty set of numbers is considered 0.
        # Here, size 0 means no elements selected, so both parts are empty.
        dp[0].add((0, 0))

        # Iterate through each number in nums
        for val in nums:
            # Iterate backwards through j (size of subsequence BEFORE adding val)
            # This ensures that when we update dp[j+1], we are using states from dp[j]
            # before dp[j] itself is potentially updated from dp[j-1].
            for j in range(2 * k, -1, -1):
                # Iterate over a copy of the set dp[j] because we might add to dp[j+1]
                # and iterating backwards in j handles overlapping states correctly.
                for or1, or2 in list(dp[j]):
                    # Option 2: Include val if current size < 2k
                    if j < 2 * k:
                        new_j = j + 1

                        # Calculate new_or1 and new_or2 for the subsequence of size new_j
                        # The state (or1, or2) in dp[j] means:
                        # if j <= k: or1 is OR(first j), or2 is 0
                        # if j > k: or1 is OR(first k), or2 is OR(last j-k)

                        if new_j <= k:
                            # Adding to the first part (new size 1 to k). Old size was j < k.
                            # new_or1 = OR(first j elements + val) = or1 | val
                            # new_or2 = OR(last max(0, new_j-k) elements) = OR(last 0 elements) = 0
                            new_or1 = or1 | val
                            new_or2 = 0
                        elif new_j == k + 1:
                            # Adding the (k+1)-th element (new size k+1). Old size was j = k.
                            # new_or1 = OR(first k elements) = or1 (from dp[k])
                            # new_or2 = OR(last new_j - k = 1 element) = val
                            new_or1 = or1
                            new_or2 = val
                        else: # new_j > k + 1 (new size k+2 to 2k). Old size was j > k.
                            # new_or1 = OR(first k elements) = or1 (from dp[j>k])
                            # new_or2 = OR(last new_j - k = j+1-k elements) = OR(last j-k elements) | val = or2 | val
                            new_or1 = or1
                            new_or2 = or2 | val

                        # Add the new state to dp[new_j]
                        dp[new_j].add((new_or1, new_or2))

        # After processing all numbers, find the maximum XOR value in dp[2*k]
        max_xor_val = 0
        # dp[2*k] contains pairs (or1, or2) where
        # or1 is the OR of the first k elements
        # or2 is the OR of the last 2k - k = k elements
        # This matches the problem definition for a sequence of size 2k.
        for or1, or2 in dp[2 * k]:
            max_xor_val = max(max_xor_val, or1 ^ or2)

        return max_xor_val