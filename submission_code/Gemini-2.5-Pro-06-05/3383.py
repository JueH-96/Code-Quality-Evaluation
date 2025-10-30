from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        """
        Calculates the maximum possible energy gain by choosing a starting magician
        and teleporting with jumps of size k.

        This problem can be viewed as finding the maximum suffix sum among k disjoint
        sequences. A dynamic programming approach is used, iterating backwards to
        efficiently calculate these suffix sums in-place.
        """
        n = len(energy)
        
        # Iterate backwards from n-k-1. For any index i >= n-k, the path starting
        # at i has only one element, so its total energy is just energy[i]. These
        # serve as the base cases for our DP.
        # For an index i < n-k, the total energy is energy[i] plus the total
        # energy from the next magician in the sequence, which is at i+k.
        for i in range(n - k - 1, -1, -1):
            energy[i] += energy[i + k]
            
        # After the loop, energy[i] stores the total energy for a path starting at i.
        # The problem allows choosing any starting magician, so the maximum possible
        # energy is the maximum value in the updated energy array.
        return max(energy)