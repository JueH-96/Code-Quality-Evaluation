from typing import List
import math

# Helper function to compute gcd
def gcd(a: int, b: int) -> int:
    """Computes the greatest common divisor of two non-negative integers."""
    while b:
        a, b = b, a % b
    return a

# Helper function to compute lcm
# Use // first to prevent potential intermediate overflow if a*b is very large,
# although Python handles arbitrary integers. This order is numerically stable.
def lcm(a: int, b: int) -> int:
    """Computes the least common multiple of two non-negative integers."""
    # LCM is typically defined for positive integers.
    # Problem constraints target[i] >= 1, so a and b derived from target will be >= 1.
    # Added checks for 0 just in case, although they should not be reached in this problem context.
    if a == 0 or b == 0:
         # Depending on context, LCM involving 0 might be 0 or undefined.
         # For positive numbers, LCM is positive. Returning 0 seems reasonable
         # if either input is 0, as 0 is a multiple of any non-zero number.
         return 0
    
    # Optimization for identity element 1
    # This optimization might be redundant given target[i] >= 1 and the initial call structure,
    # but it's generally correct for an LCM function.
    if a == 1: return b
    if b == 1: return a

    # Standard LCM formula: lcm(a, b) = |a * b| / gcd(a, b)
    # For positive a, b: lcm(a, b) = (a * b) / gcd(a, b)
    # Using (a // gcd(a, b)) * b is numerically better when a*b might overflow
    # standard integer types, though Python handles arbitrary size.
    return (a // gcd(a, b)) * b


class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        """
        Calculates the minimum operations (increments) on nums elements
        so that each element in target has at least one multiple in nums.

        Args:
            nums: A list of integers.
            target: A list of integers.

        Returns:
            The minimum number of operations required.
        """

        # Function to get all distinct LCMs of non-empty subsets of target
        # These are the candidate values we might increment nums elements to become multiples of.
        def get_distinct_lcms(target_list):
            lcms = set()
            n = len(target_list)

            # Recursive helper to generate subsets and calculate their LCM
            # index: current index in target_list being considered
            # current_lcm_val: the LCM of elements included in the current subset being built
            # is_empty_subset: boolean, true if the current subset being built is empty
            def generate_subsets_and_lcm(index, current_lcm_val, is_empty_subset):
                # Base case: finished considering all target elements
                if index == n:
                    # If the subset formed is not empty, add its calculated LCM to the set of distinct LCMs
                    if not is_empty_subset:
                        lcms.add(current_lcm_val)
                    return

                # Option 1: Exclude target_list[index] from the current subset
                # Recurse to consider the next index, keeping the current LCM and empty status.
                generate_subsets_and_lcm(index + 1, current_lcm_val, is_empty_subset)

                # Option 2: Include target_list[index] in the current subset
                # Calculate the new LCM. If the subset was previously empty, the new LCM is just target_list[index].
                # Otherwise, it's lcm(current_lcm_val, target_list[index]).
                next_lcm_val = lcm(current_lcm_val, target_list[index]) if not is_empty_subset else target_list[index]

                # Recurse with the next index and the updated LCM value. The subset is definitely not empty now.
                generate_subsets_and_lcm(index + 1, next_lcm_val, False)

            # Start the recursive generation process.
            # We begin by considering the first element (index 0) with an conceptually empty subset.
            # The identity element for LCM is 1. We pass `is_empty_subset=True` initially.
            generate_subsets_and_lcm(0, 1, True)

            # Convert the set of unique LCMs to a list and sort it. Sorting is not strictly needed for correctness
            # but can help in debugging or analysis.
            return sorted(list(lcms))

        # Function to calculate the minimum cost (increments) to make number n a multiple of L
        def cost_to_make_multiple(n, L):
            """Calculates the minimum increments to make n a multiple of L."""
            # If L is non-positive, it's impossible to make a positive number a multiple.
            # Problem constraints ensure target[i] >= 1, so L derived from target will be >= 1.
            # This check is mostly for robustness.
            if L <= 0: return float('inf')
            
            # If n is already a multiple of L, no increments are needed.
            if n % L == 0: return 0

            # Otherwise, we need to increment n to the smallest multiple of L that is greater than or equal to n.
            # This value is calculated as ceil(n / L) * L.
            # Using integer division: ((n + L - 1) // L) * L
            smallest_multiple_ge_n = ((n + L - 1) // L) * L

            # The cost is the difference between the target multiple we want to reach and the original number n.
            return smallest_multiple_ge_n - n

        # Function to get the bitmask representing which targets are covered by a number
        # that is made a multiple of L.
        def get_cover_mask(L, target_list):
            """Returns a bitmask where the i-th bit is set if L is a multiple of target_list[i]."""
            mask = 0
            for i in range(len(target_list)):
                # If L is a multiple of target_list[i], then any number that is a multiple of L
                # is also a multiple of target_list[i]. This means target_list[i] is covered.
                if L % target_list[i] == 0:
                    # Set the i-th bit in the mask (using 0-based indexing for target_list).
                    mask |= (1 << i)
            return mask

        # k is the number of target elements. This determines the size of our DP state space.
        k = len(target)
        # num_masks is the total number of possible subset masks for k elements (2^k).
        num_masks = 1 << k

        # dp[mask] will store the minimum cost to satisfy the conditions
        # for the subset of targets represented by the bitmask 'mask'.
        # Initialize the DP table with infinity for all states, as initially, no targets are covered
        # with minimal cost known.
        # dp[0] represents the state where no targets are covered, which requires 0 cost initially.
        dp = [float('inf')] * num_masks
        dp[0] = 0

        # Get all distinct LCMs of non-empty subsets of the target elements.
        # These distinct LCM values represent the possible "goals" we might set for a number in nums
        # when incrementing it – making it a multiple of such an LCM value.
        distinct_lcms = get_distinct_lcms(target)

        # This is the core dynamic programming part.
        # Iterate through each number 'n' in the input array 'nums'.
        # Each number in 'nums' provides an opportunity to improve the minimum cost
        # to cover various subsets of targets.
        for n in nums:
            # For the current number 'n', consider the possibility of making it a multiple
            # of each potential LCM value 'L' that we identified from target subsets.
            for L in distinct_lcms:
                # Calculate the minimum cost required to increment 'n' so it becomes a multiple of 'L'.
                c = cost_to_make_multiple(n, L)

                # Determine exactly which targets (represented by a bitmask) are covered
                # if the current number 'n' is made a multiple of this specific LCM value 'L'.
                covered_mask = get_cover_mask(L, target)

                # Now, update the DP table based on the potential contribution of using the current number 'n'.
                # We iterate through all possible current states (masks) that are already reachable.
                # The iteration order for 'mask' is crucial here: decreasing order.
                # Iterating downwards ensures that when we calculate dp[mask | covered_mask], the value dp[mask]
                # used on the right side of the equation is the minimum cost to reach state 'mask'
                # using *only* the numbers from 'nums' processed *before* the current number 'n',
                # or using the current number 'n' but for covering a completely different set of targets
                # (i.e., where 'mask' does not contain any bits from 'covered_mask' that were *newly* set
                # by the current operation). This prevents using a cost improvement from the current
                # (n, L) operation to immediately benefit a larger mask calculation within the same pass
            
                for mask in range(num_masks - 1, -1, -1):
                    # If the current state 'mask' is reachable (i.e., its cost is not infinity)
                    if dp[mask] != float('inf'):
                        # The new state 'mask | covered_mask' can be reached by
                        # achieving state 'mask' with minimum cost dp[mask], AND then using
                        # the current number 'n' to cover the targets in 'covered_mask' with cost 'c'.
                        # We update the minimum cost to reach the combined state 'mask | covered_mask'
                        # by taking the minimum of its current value and the cost of this new path.
                        dp[mask | covered_mask] = min(dp[mask | covered_mask], dp[mask] + c)

        # After iterating through all numbers in 'nums' and all relevant LCMs,
        # the entry dp[num_masks - 1] will hold the minimum total cost to reach the state
        # where all bits are set, meaning all target elements have at least one multiple
        # in the modified 'nums' array.
        # If it were impossible to cover all targets, this value would theoretically remain float('inf'),
        # but given the constraints (target.length <= nums.length), it should always be possible.
        return dp[num_masks - 1]