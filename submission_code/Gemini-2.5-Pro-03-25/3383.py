import math
from typing import List

class Solution:
    """
    Solves the maximum energy problem using dynamic programming.
    The approach utilizes in-place modification of the energy array
    to store the cumulative energy for paths starting at each index.
    """
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        """
        Calculates the maximum possible energy gain by starting from any magician
        and jumping k steps until the end of the sequence.

        Args:
            energy: A list of integers representing the energy gain from each magician.
                    This list will be modified in-place.
            k: An integer representing the jump distance.

        Returns:
            The maximum possible energy gain achievable.
        """
        n = len(energy)

        # The problem asks for the maximum sum of energies collected along paths
        # starting from any index `i` and jumping by `k` until the end.
        # Path from i: energy[i] + energy[i+k] + energy[i+2k] + ...

        # We can use a dynamic programming approach. Let S[i] be the total
        # energy gained starting from magician i.
        # The recurrence relation is:
        # S[i] = energy[i]                   if i + k >= n
        # S[i] = energy[i] + S[i + k]        if i + k < n
        # We want to find the maximum S[i] for 0 <= i < n.

        # We can compute these S[i] values efficiently by iterating backwards.
        # By iterating from i = n - 1 down to 0, when we compute S[i],
        # the value S[i + k] (if needed) has already been computed.
        # We can store the results S[i] back into the energy array itself
        # to achieve O(1) space complexity (excluding input storage),
        # assuming modification of the input array is allowed.

        # Iterate backwards starting from the index n - k - 1. This is the largest index 'i'
        # such that i + k is still within the array bounds (i + k < n).
        # The indices from n - k to n - 1 do not need updates in this loop because
        # for them, i + k >= n. Their initial energy[i] value already represents
        # the total energy S[i] for paths starting there (which consist of only one step).
        # The range(start, stop, step) function goes up to, but does not include, 'stop'.
        # So range(n - k - 1, -1, -1) covers indices from n - k - 1 down to 0.
        for i in range(n - k - 1, -1, -1):
            # Update energy[i] to store the total path sum S[i].
            # energy[i + k] at this point already holds the computed S[i + k]
            # because of the backward iteration order.
            energy[i] += energy[i + k]

        # After the loop completes, the `energy` array effectively stores the total energy
        # S[i] for each possible starting position i (0 <= i < n).
        # The maximum value in this modified array represents the overall maximum
        # possible energy gain among all possible starting points.
        # The built-in `max()` function finds this maximum value efficiently.
        # It correctly handles cases with negative energy values.
        # The constraint n >= 1 ensures the array is not empty, so max() is safe.
        return max(energy)