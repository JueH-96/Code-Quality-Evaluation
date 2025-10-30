from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of adjacent swaps to make the permutation semi-ordered.
        A semi-ordered permutation has 1 at the first position (index 0)
        and n at the last position (index n-1).

        The minimum number of adjacent swaps to move an element from index i to index j
        is abs(i - j).

        To make the permutation semi-ordered, we need to move the element 1
        to index 0 and the element n (where n is the length of the list)
        to index n-1.

        Let idx_1 be the current index of 1, and idx_n be the current index of n.

        Minimum swaps to move 1 to index 0 is idx_1 (move left).
        Minimum swaps to move n to index n-1 is (n-1) - idx_n (move right).

        The total minimum swaps is the sum of these two independent movements,
        unless the movement of one element helps the movement of the other.

        This "help" or overlap happens when 1 is initially to the right of n
        (idx_1 > idx_n). In this case, the final configuration requires 1
        to be to the left of n. The process of reversing their relative order
        costs exactly one adjacent swap (or the net effect of a sequence of swaps
        that achieves the same relative reversal when they are adjacent).
        This single swap contributes to moving 1 left AND moving n right.
        This shared swap is counted in both the 'idx_1' swaps for moving 1 left
        and the '(n-1) - idx_n' swaps for moving n right. To avoid double-counting
        this overlap, we subtract 1 from the total sum when idx_1 > idx_n.

        Args:
            nums: A 0-indexed permutation of n integers.

        Returns:
            The minimum number of operations (adjacent swaps).
        """
        n = len(nums)
        
        # Find the current index of the number 1
        # The list.index() method finds the first occurrence of the value.
        # Since nums is a permutation, 1 and n exist exactly once.
        idx_1 = nums.index(1)
        
        # Find the current index of the number n
        idx_n = nums.index(n)

        # Calculate the minimum swaps needed to move 1 to the front (index 0).
        # This is the number of elements currently to its left that it needs to pass.
        swaps_to_move_1_to_start = idx_1
        
        # Calculate the minimum swaps needed to move n to the end (index n-1).
        # This is the number of elements currently to its right that it needs to pass.
        swaps_to_move_n_to_end = (n - 1) - idx_n
        
        # The initial total is the sum of swaps for each movement assuming they don't interfere.
        total_swaps = swaps_to_move_1_to_start + swaps_to_move_n_to_end
        
        # Check for interference: If 1 is initially to the right of n (idx_1 > idx_n),
        # then moving 1 left and n right will eventually cause them to swap positions
        # relative to each other. This relative order reversal is a single operation
        # (when they are adjacent) that contributes to both desired movements.
        # This shared swap is implicitly included in the 'swaps_to_move_1_to_start'
        # and 'swaps_to_move_n_to_end' calculations. Therefore, we subtract 1
        # to avoid double-counting this specific swap.
        if idx_1 > idx_n:
            total_swaps -= 1
            
        return total_swaps