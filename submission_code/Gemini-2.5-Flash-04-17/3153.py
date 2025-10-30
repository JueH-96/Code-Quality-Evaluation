from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # Max value 10^9 is less than 2^30 (1073741824).
        # So, bits from 0 up to 29 are relevant.
        # Using 31 bits (0 to 30) to be safe, covering up to 2^30.
        MAX_BITS = 31

        # Step 1: Calculate bit counts
        # bit_counts[p] stores the total number of elements in the array
        # that have the p-th bit set in the initial array.
        # This count is invariant under the operation.
        bit_counts = [0] * MAX_BITS
        for num in nums:
            for p in range(MAX_BITS):
                if (num >> p) & 1:
                    bit_counts[p] += 1

        # Step 2: Construct the k largest potential values
        # The operation allows redistributing the '1' bits at each position.
        # For each bit position p, there are bit_counts[p] ones in total across the n elements.
        # To maximize the sum of squares of k elements, we should make the k elements
        # as large as possible. This can be achieved by consolidating the available
        # '1' bits for each position into the elements that will end up being the largest.
        # Specifically, we can arrange the final array such that for each bit p,
        # the first bit_counts[p] elements (when sorted in descending order) have bit p set.
        # The k largest elements will be the first k elements in this sorted arrangement.
        # The i-th element (0-indexed, i from 0 to k-1) among the top k
        # corresponds to the (i+1)-th element in the full sorted array.
        # This element will have bit p set if its rank (i+1) is less than or equal to
        # the total count of 1s available for bit p (bit_counts[p]).
        # i.e., i + 1 <= bit_counts[p], which means i < bit_counts[p].

        final_k_values = [0] * k

        # Iterate through bit positions from 0 upwards
        for p in range(MAX_BITS):
            # The number of elements among the top k that will have bit p set
            # is the minimum of k (the number of elements we are choosing)
            # and bit_counts[p] ( the total number of elements that can have bit p set).
            num_elements_to_set_bit_p = min(k, bit_counts[p])

            # Assign bit p to the first num_elements_to_set_bit_p values
            # among the k values we are building.
            # These correspond to the top num_elements_to_set_bit_p values
            # among the k largest.
            for i in range(num_elements_to_set_bit_p):
                 final_k_values[i] |= (1 << p)

        # Step 3: Calculate the sum of squares modulo MOD
        sum_of_squares = 0
        for val in final_k_values:
            # Calculate (val * val) % MOD safely for large val
            square = (val % MOD) * (val % MOD) % MOD
            sum_of_squares = (sum_of_squares + square) % MOD

        return sum_of_squares