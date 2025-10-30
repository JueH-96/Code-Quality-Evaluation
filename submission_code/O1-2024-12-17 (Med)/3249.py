class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        We want the XOR of all final array elements to be k. Flipping bits in different
        positions is independent from each other (because each flip changes exactly one
        bit in exactly one element). Therefore, we can handle each bit position separately.

        -- Key Idea --
        Let c_1[i] be the count of elements that have bit i set (i.e., the number of '1's
        in the i-th bit position across all nums).

        The XOR in the i-th bit of the final array is determined by the parity of c_1[i].
        We want that parity to match the i-th bit of k (denote that bit b_i).

        If c_1[i] % 2 == b_i, the parity is already correct. No flips are needed for that bit.
        Otherwise, exactly one flip in that bit position will fix the parity. We can always
        do this flip (by flipping a '1' to '0' if any exist, or a '0' to '1' otherwise)
        as long as we have at least one array element, which is guaranteed.

        Hence, for each bit i:
            cost_i = 0 if (c_1[i] % 2 == b_i)
            cost_i = 1 otherwise

        The total minimum operations is the sum of cost_i over all relevant bit positions.
        Since nums[i] and k are up to 10^6, we only need to check up to 20 bits (0..20).
        """
        # We only need ~20 bits because 10^6 < 2^20
        MAX_BITS = 20

        # Count how many elements have each bit i set
        count_ones = [0] * (MAX_BITS + 1)
        for num in nums:
            for i in range(MAX_BITS + 1):
                if (num >> i) & 1:
                    count_ones[i] += 1

        # Compute cost by checking parity of each bit
        operations = 0
        for i in range(MAX_BITS + 1):
            desired_bit = (k >> i) & 1   # bit i of k
            if (count_ones[i] % 2) != desired_bit:
                operations += 1

        return operations