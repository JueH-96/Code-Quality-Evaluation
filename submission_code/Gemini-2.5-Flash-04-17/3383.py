from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        """
        Calculates the maximum possible energy gained by choosing a starting
        magician and jumping k steps repeatedly.

        Args:
            energy: A list of integers representing the energy gained from
                    each magician. energy[i] is the energy from magician i.
            k: An integer representing the jump step. After absorbing energy
               from magician i, the next magician is i + k.

        Returns:
            The maximum possible energy gain among all possible starting points.
        """
        n = len(energy)

        # We can use dynamic programming to calculate the total energy
        # for a path starting at each index i.
        # Let path_sum[i] be the total energy gained by starting at index i
        # and following the k-step jumps.
        # The path from index i is i, i+k, i+2k, ..., as long as the index is < n.
        # The total energy is the sum of energy at these indices.
        # We can observe a recurrence relation:
        # path_sum[i] = energy[i] + energy[i+k] + energy[i+2k] + ...
        # If i + k < n, then the path from i is energy[i] + (path from i+k).
        # So, path_sum[i] = energy[i] + path_sum[i+k] (if i+k < n).
        # If i + k >= n, the path ends at i. So, path_sum[i] = energy[i].

        # This recurrence relation suggests a backward calculation.
        # We can compute path_sum[i] values by iterating from n-1 down to 0.
        # When we are at index i, if i + k < n, we need path_sum[i+k].
        # Since i+k > i, the value path_sum[i+k] will have already been computed
        # if we iterate downwards.
        # We can store the path_sum[i] values directly in the energy array
        # to save space (O(1) auxiliary space).

        # Iterate backwards from the end of the array.
        # The loop processes indices from n-1 down to 0.
        for i in range(n - 1, -1, -1):
            # Check if jumping k steps forward from index i results in a valid index.
            # If i + k is within the array bounds [0, n-1], then the path starting at i
            # includes the energy at index i plus the total energy from the path
            # starting at index i+k.
            if i + k < n:
                # Add the pre-calculated total energy from the next jump point.
                # At this point in the backward iteration, energy[i+k] already stores
                # the total energy for the path starting from i+k (i.e., path_sum[i+k]).
                energy[i] += energy[i + k]
            # If i + k >= n, the path starting at index i includes only energy[i]
            # because the next jump is out of bounds. In this case, energy[i]
            # already holds its original value, which is the correct path_sum[i].
            # So, no update is needed for this case within the if condition.

        # After the loop finishes, energy[i] for each index i (from 0 to n-1)
        # contains the total energy gained by starting the journey at index i
        # and following the k-step jumps until the end of the array.
        # The problem asks for the maximum possible energy gain, which is the
        # maximum value among all these possible total energies.

        # The maximum possible energy is simply the maximum value present
        # in the modified energy array after the DP calculation.
        # The constraints guarantee 1 <= energy.length <= 10^5, which implies n >= 1.
        # The constraint 1 <= k <= energy.length - 1 implies energy.length >= 2, so n >= 2.
        # Thus, the energy array is guaranteed to have at least 2 elements,
        # and calling max() on it is safe and will return a value.
        return max(energy)