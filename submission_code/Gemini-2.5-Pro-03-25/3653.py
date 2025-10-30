import math
from typing import List

# Define the class Solution as required by the starter code format.
class Solution:
    # Define the method maxSubarraySum within the Solution class.
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Finds the maximum sum of a subarray of nums such that the size of the subarray is divisible by k.

        The approach uses prefix sums and dynamic programming based on remainders modulo k.
        Let P[x] be the prefix sum of the first x elements (P[x] = nums[0] + ... + nums[x-1]), with P[0] = 0.
        The sum of a subarray nums[i:j+1] (from index i to j, inclusive) is P[j+1] - P[i].
        The length of this subarray is (j + 1) - i.
        We require the length to be divisible by k, which means (j + 1 - i) % k == 0.
        This condition is equivalent to (j + 1) % k == i % k.

        Let x = j + 1 and y = i. We are looking for the maximum value of P[x] - P[y]
        subject to x % k == y % k and 0 <= y < x <= n, where n is the length of nums.

        We iterate through the array, calculating prefix sums P[x] for x from 1 to n.
        For each P[x], we need to find the minimum P[y] encountered so far among all y < x
        such that y % k == x % k. Let this minimum value be min_P_y.
        The sum of the subarray corresponding to this pair (y, x) is P[x] - min_P_y.
        The length of this subarray is x - y, which is guaranteed to be divisible by k.
        We maintain the overall maximum sum found across all possible ending indices (j or x-1).

        To efficiently find the minimum P[y] for each required remainder, we use an array `min_prefix_sum` of size k.
        `min_prefix_sum[r]` stores the minimum prefix sum P[y] seen so far where the index y satisfies y % k == r.


        Args:
            nums: A list of integers.
            k: An integer representing the required divisor for the subarray length.

        Returns:
            The maximum sum of a subarray whose length is divisible by k.
        """
        n = len(nums) # Get the length of the input array.

        # min_prefix_sum is an array of size k.
        # min_prefix_sum[r] stores the minimum prefix sum P[y] encountered so far
        # such that the index y satisfies y % k == r.
        # Initialize all entries to infinity, as we are looking for the minimum value.
        min_prefix_sum = [math.inf] * k
        
        # Base case: The prefix sum before the first element is P[0] = 0.
        # The index associated with P[0] is y=0.
        # The remainder is 0 % k = 0.
        # So, the minimum prefix sum for remainder 0 starts at 0.
        min_prefix_sum[0] = 0

        # Initialize the maximum sum found so far to negative infinity.
        # This ensures that any valid sum (including negative or zero) will be correctly captured as the maximum.
        max_s = -math.inf
        
        # current_prefix_sum stores the prefix sum P[x] as we iterate through the array.
        # Initialize to P[0] = 0 before the loop starts.
        current_prefix_sum = 0

        # Iterate through the input array nums using index i from 0 to n-1.
        for i in range(n):
            # Update the current prefix sum by adding the current element nums[i].
            # After this operation, current_prefix_sum holds the value of P[i+1].
            current_prefix_sum += nums[i]

            # Let x = i + 1. This is the 1-based index corresponding to the prefix sum P[x].
            # Example: when i=0, x=1, current_prefix_sum = P[1] = nums[0].
            x = i + 1
            
            # Calculate the remainder of the index x when divided by k.
            remainder = x % k

            # We are looking for a previous index y (where 0 <= y < x) such that y % k == remainder.
            # If such a y exists, the subarray nums[y...i] (which is nums[y : i+1]) has length x - y.
            # Since x % k == y % k, the length x - y must be divisible by k.
            # The sum of this subarray is P[x] - P[y].
            # To maximize this sum for the current P[x], we need the minimum P[y] among all valid y seen so far.

            # Retrieve the minimum prefix sum P[y] that has been recorded for this remainder `remainder`.
            min_prev_pfs = min_prefix_sum[remainder]

            # Check if we have encountered a valid previous prefix sum P[y] for this remainder.
            # If min_prev_pfs is not infinity, it means such a y exists (y < x and y % k == remainder).
            if min_prev_pfs != math.inf:
                # Calculate the sum of the subarray ending at index i with length divisible by k.
                # This sum is P[x] - P[y] = current_prefix_sum - min_prev_pfs.
                current_subarray_sum = current_prefix_sum - min_prev_pfs
                
                # Update the overall maximum sum found if the current subarray sum is greater.
                max_s = max(max_s, current_subarray_sum)

            # Update the minimum prefix sum recorded for the current remainder `remainder`.
            # The current prefix sum P[x] (stored in `current_prefix_sum`) corresponds to index x.
            # This P[x] value might be the new minimum for indices with remainder `x % k`.
            # We update if `current_prefix_sum` is smaller than the currently stored minimum for this remainder.
            min_prefix_sum[remainder] = min(min_prefix_sum[remainder], current_prefix_sum)

        # The constraints (n >= k >= 1) guarantee that at least one subarray with length divisible by k exists.
        # For example, the subarray nums[0...k-1] has length k. Its sum is P[k] - P[0].
        # The algorithm considers this sum when x = k (i.e., when i = k-1). At this point, remainder = k % k = 0.
        # It looks up min_prefix_sum[0], which was initialized to 0 (P[0]).
        # The sum P[k] - P[0] is calculated and compared with max_s.
        # Therefore, max_s will always be updated from its initial -infinity value to a valid sum.
        
        # Return the final maximum sum found.
        return max_s