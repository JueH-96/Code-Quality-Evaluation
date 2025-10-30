class Solution:
    def minimumSteps(self, s: str) -> int:
        """
        Calculates the minimum number of adjacent swaps to group all '1's
        to the right and all '0's to the left in a binary string.

        This problem is equivalent to sorting the binary string using
        adjacent swaps, which has a cost equal to the number of inversions.
        An inversion here is a pair of indices (i, j) such that i < j
        and s[i] = '1' and s[j] = '0'.

        We can count these inversions by iterating through the string
        from left to right. We maintain a count of '1's encountered so far.
        When we encounter a '0', every '1' seen before it must be swapped
        past this '0' to achieve the sorted state. The number of such '1's
        is our current 'one_count'. We add this to the total swaps.

        Args:
            s: A binary string (0s and 1s).

        Returns:
            The minimum number of swaps (steps) as an integer.
        """
        total_swaps = 0  # Initialize the total count of adjacent swaps
        one_count = 0    # Counter for the number of '1's encountered from the left

        # Iterate through the string from left to right
        for char in s:
            if char == '1':
                # If we see a '1', increment the count of '1's seen so far.
                # This '1' will contribute to the swap count when we later
                # encounter '0's that are currently to its right.
                one_count += 1
            elif char == '0':
                # If we see a '0', it means this '0' is currently to the right
                # of all `one_count` '1's encountered so far.
                # To reach the sorted state where all '0's are left of all '1's,
                # this '0' needs to be effectively moved to the left past all
                # those `one_count` '1's. The number of adjacent swaps required
                # for this '0' to move past all those '1's is exactly `one_count`.
                # Add this count to the total swaps.
                total_swaps += one_count

        return total_swaps