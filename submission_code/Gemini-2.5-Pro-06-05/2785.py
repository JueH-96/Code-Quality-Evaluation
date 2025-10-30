from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of adjacent swaps to make a permutation semi-ordered.

        A semi-ordered permutation has 1 as the first element and n as the last element.
        The minimum number of adjacent swaps to move an element is its distance from the target position.

        Args:
            nums: A 0-indexed permutation of n integers.

        Returns:
            The minimum number of operations required.
        """
        n = len(nums)
        
        # Find the 0-indexed positions of 1 and n.
        # list.index() performs a linear scan, which is efficient for n <= 50.
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # The number of adjacent swaps to move element 1 to the first position (index 0)
        # is its current index, `pos1`.
        #
        # The number of adjacent swaps to move element n to the last position (index n-1)
        # is the distance from its current position, `(n - 1) - posn`.
        #
        # If 1 starts to the left of n (pos1 < posn), the paths they take to their
        # destinations do not cross. The total number of swaps is the sum of the
        # swaps required for each.
        #
        # If 1 starts to the right of n (pos1 > posn), their paths must cross.
        # The single swap where 1 and n move past each other contributes to both goals.
        # It moves 1 one step closer to the front and n one step closer to the back.
        # The sum `pos1 + (n - 1 - posn)` counts this shared move twice,
        # so we have overcounted by one. We correct this by subtracting 1.
        
        # Start with the sum of swaps as if movements were independent.
        total_swaps = pos1 + (n - 1 - posn)
        
        # Adjust if 1 is to the right of n, as one swap is shared.
        if pos1 > posn:
            total_swaps -= 1
            
        return total_swaps