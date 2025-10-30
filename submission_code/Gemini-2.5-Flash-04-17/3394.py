class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        Constructs a strictly increasing array of n positive integers nums
        where the bitwise AND of all elements is x. Returns the minimum
        possible value of nums[n - 1].

        Args:
            n: The size of the array (1 <= n <= 10^8).
            x: The required bitwise AND result (1 <= x <= 10^8).

        Returns:
            The minimum possible value of the last element nums[n - 1].
        """
        # Let nums = [a_0, a_1, ..., a_{n-1}].
        # Conditions: a_i > 0 for all i, a_0 < a_1 < ... < a_{n-1}, and a_0 & ... & a_{n-1} = x.
        # The bitwise AND being x implies that for any bit position, if the bit
        # is set in x, it must be set in all a_i. Thus, a_i must be of the form x | k_i,
        # where k_i has bits set only in positions where x has a 0 (i.e., k_i & x == 0).

        # The condition a_0 < a_1 < ... < a_{n-1} implies x|k_0 < x|k_1 < ... < x|k_{n-1}.
        # Since k_i & x == 0, this is equivalent to k_0 < k_1 < ... < k_{n-1}.
        # We want to minimize a_{n-1} = x | k_{n-1}, which is minimized by minimizing k_{n-1}.

        # Let p_0 < p_1 < p_2 < ... be the bit positions where x has a 0. These are the "free" bit positions.
        # Any number k formed using only these free bits can be represented by mapping a standard non-negative
        # integer m (binary ...b_2 b_1 b_0) to k. The mapping function, map(m), sets the p_i-th bit in k
        # if the i-th bit b_i in m is 1. map(m) = sum_{i=0} ((m >> i) & 1) * 2^{p_i}.

        # The sequence k_0, k_1, ..., k_{n-1} must be strictly increasing and k_i & x == 0.
        # Also, a_i = x | k_i must be positive.
        # Since the constraint is 1 <= x <= 10^8, x is always positive.
        # The elements a_i = x | k_i must be positive. Since x > 0, x | k_i >= x > 0 for any k_i >= 0.
        # To minimize k_{n-1} while keeping k_0 < ... < k_{n-1} strictly increasing,
        # we choose the smallest possible non-negative k values formed by free bits.
        # These are map(0), map(1), map(2), ... in increasing order.
        # The smallest n such values in increasing order are map(0), map(1), ..., map(n-1).
        # So, we can set k_i = map(i) for i = 0, ..., n-1.
        # The array nums becomes [x|map(0), x|map(1), ..., x|map(n-1)]. This sequence is strictly increasing and positive.
        # The last element is a_{n-1} = x | k_{n-1} = x | map(n-1). We need to calculate x | map(n-1).
        # This means we take the binary representation of n-1 and use its bits to set the corresponding
        # free bit positions in x.

        ans = x # Initialize the result with x. Bits set in x are always set.

        num_to_map = n - 1 # The number whose binary representation dictates which free bits of x are set in k_{n-1}.

        # `bit_idx_in_num_to_map` is the index of the bit we are reading from `num_to_map` (0-th, 1st, 2nd...).
        # This also corresponds to the index of the free bit position in x we are looking for (0th free bit, 1st free bit, etc.).
        bit_idx_in_num_to_map = 0

        # `j` iterates through potential bit positions in the result `ans`.
        # We search for free bit positions in x by iterating `j` upwards.
        j = 0

        # We loop through physical bit positions `j = 0, 1, 2, ...`.
        # In each iteration, we check if position `j` is free in `x`.
        # If it is free, this `j` is the `bit_idx_in_num_to_map`-th free bit we have encountered.
        # We then check the corresponding bit in `num_to_map`.
        # We will break when we have processed all required bits from `num_to_map`.
        # This happens when `(num_to_map >> bit_idx_in_num_to_map)` becomes 0.

        while True:
            # Check if the `j`-th bit position in x is free (0).
            if ((x >> j) & 1) == 0:
                # This position `j` is a free bit position in x. It is the `bit_idx_in_num_to_map`-th free bit we have encountered.

                # Check if we have processed all required bits of `num_to_map`.
                # `(num_to_map >> bit_idx_in_num_to_map)` gives the remaining part of `num_to_map` after right-shifting.
                # If this is 0, it means all bits in `num_to_map` at or above `bit_idx_in_num_to_map` are 0.
                # We have used enough free bit positions (up to `bit_idx_in_num_to_map - 1`) to map the set bits of `num_to_map`.
                # If `num_to_map` was 0 initially (n=1), `(0 >> 0) == 0`. The loop breaks immediately, returning `x`.
                if (num_to_map >> bit_idx_in_num_to_map) == 0:
                    break # All relevant bits of num_to_map have been processed or are zero.

                # If the `bit_idx_in_num_to_map`-th bit of `num_to_map` is 1, we must set this free bit position `j` in `ans`.
                if ((num_to_map >> bit_idx_in_num_to_map) & 1) == 1:
                    ans = ans | (1 << j)

                # We have used this free bit position `j` corresponding to the `bit_idx_in_num_to_map`-th bit of `num_to_map`.
                # Move to consider the next bit index in `num_to_map` for the next free bit position we find in `x`.
                bit_idx_in_num_to_map += 1

            # Move to the next physical bit position `j` to check in `x`.
            j += 1

        return ans