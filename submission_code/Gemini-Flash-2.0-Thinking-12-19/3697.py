from typing import List
import math

# Helper functions outside the class for clarity, or could be static methods
def gcd(a, b):
    """Calculates the greatest common divisor of two integers."""
    # math.gcd handles arbitrary large integers in Python 3.8+
    return math.gcd(a, b)

def lcm(a, b):
    """Calculates the least common multiple of two non-zero integers."""
    # Since target[i] >= 1, LCM of subset of target will be >= 1.
    # We assume a and b are positive integers here.
    # Use the formula lcm(a, b) = (a * b) // gcd(a, b)
    # Python 3 handles arbitrary precision integers, so (a * b) is fine.
    # In languages with fixed-size integers, one might use (a // gcd(a, b)) * b to avoid overflow
    # if the product a * b exceeds the integer limit, but a // gcd(a, b) fits,
    # and (a // gcd(a, b)) * b also fits in the result type.
    # The context ensures a and b are from target or intermediate LCMs, so they are >= 1.
    return (a * b) // gcd(a, b)

def list_lcm(numbers):
    """Calculates the least common multiple of a list of positive integers."""
    # Assumes numbers is not empty and contains positive integers.
    # In our DP, s_mask >= 1 guarantees a non-empty subset from target which has elements >= 1.
    if not numbers:
         # Should not happen with s_mask >= 1 and target length >= 1
         return 1 
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        m = len(target)
        # dp[mask] = minimum cost to cover the set of targets represented by mask
        # mask is a bitmask where the j-th bit is 1 if target[j] is covered.
        dp = [float('inf')] * (1 << m)
        dp[0] = 0 # Base case: 0 cost to cover no targets

        # Create a list of target values for easy access by index j
        target_list = target # Alias for clarity

        for num in nums:
            # When considering a number `num` from the input array `nums`,
            # we explore how using this number (by incrementing it) can help cover
            # additional target requirements.
            # We iterate through all previously reachable DP states (`prev_mask`).
            # For each `prev_mask`, we consider incrementing `num` to satisfy
            # requirements for a subset `S` of target values, which might transition
            # the state from `prev_mask` to `prev_mask | covered_m`.

            # Iterate through masks in reverse order to ensure we use dp values from
            # the previous iteration (before processing the current 'num').
            # If we iterate upwards, we might use a `dp[prev_mask]` that was
            # already improved by the current `num`'s contribution to `prev_mask` itself,
            # effectively using the current `num` multiple times.
            # Iterating downwards ensures `dp[prev_mask]` was finalized before considering
            # the current `num`'s effect on `prev_mask`.
            for prev_mask in range((1 << m) - 1, -1, -1):
                if dp[prev_mask] == float('inf'):
                    continue # Cannot reach this state

                # Consider all possible non-empty subsets of targets that
                # this number `num` could be used to satisfy.
                # Represent the subset by a mask `s_mask`.
                # By incrementing `num` to be a multiple of `L = lcm(S)`, we ensure
                # it is a multiple of every target value in `S`.
                for s_mask in range(1, 1 << m):
                    # Build the subset of target values for this mask
                    subset_target_values = [target_list[j] for j in range(m) if (s_mask >> j) & 1]

                    # Calculate the least common multiple (L) of this subset
                    # Since target[i] >= 1 and s_mask >= 1, L = list_lcm(...) will be >= 1.
                    L = list_lcm(subset_target_values)

                    # Calculate the smallest multiple of L that is >= num
                    # Using integer division: target_multiple = ((num + L - 1) // L) * L
                    # This works correctly for positive num and positive L.
                    # If num is 0, the smallest multiple >= 0 is 0.
                    # If num > 0 and num % L == 0, (num // L) * L = num.
                    # If num > 0 and num % L != 0, (num // L + 1) * L.
                    
                    target_multiple = ((num + L - 1) // L) * L

                    # The cost of this operation is the increment amount
                    cost = target_multiple - num

                    # Determine which target values are actually covered by target_multiple.
                    # A multiple of L is a multiple of all elements in the original subset S.
                    # It might also be a multiple of other target values not in S.
                    covered_m = 0
                    for j in range(m):
                        if target_multiple % target_list[j] == 0:
                            covered_m |= (1 << j)

                    # Update the DP table:
                    # If we could previously cover `prev_mask` with cost `dp[prev_mask]`,
                    # we can now cover `prev_mask | covered_m` with cost `dp[prev_mask] + cost`.
                    dp[prev_mask | covered_m] = min(dp[prev_mask | covered_m], dp[prev_mask] + cost)

        # The minimum cost to cover all targets is stored in dp[(1 << m) - 1]
        return dp[(1 << m) - 1]