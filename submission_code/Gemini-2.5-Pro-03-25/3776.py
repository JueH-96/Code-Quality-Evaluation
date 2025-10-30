import sys
# Setting a higher recursion depth might be necessary for larger inputs (n > ~500),
# depending on the specific test cases and recursion patterns.
# sys.setrecursionlimit(2000) 

from functools import lru_cache
from typing import List, Tuple

# Enclose the solution within the provided class structure.
class Solution:
    """
    Solves the problem of finding the minimum cost to remove all elements from an array
    using specified operations involving the first three elements.
    Uses memoized recursion (dynamic programming) to avoid redundant calculations.
    The state of the recursion is represented by the tuple of remaining numbers.
    """
    def minCost(self, nums: List[int]) -> int:
        """
        Calculates the minimum cost to remove all elements from the list `nums`.

        The core logic is implemented in the recursive helper function `solve`,
        which uses memoization (@lru_cache) to store and retrieve results for
        previously encountered states (subproblems).

        Args:
            nums: A list of integers representing the initial array.

        Returns:
            The minimum cost required to remove all elements according to the rules.
        """
        
        # Convert the input list to a tuple. Tuples are immutable and hashable,
        # which is required for using them as keys in the memoization cache (lru_cache).
        # This allows the @lru_cache decorator to effectively store results for subproblems.
        nums_tuple = tuple(nums)

        # Define the recursive solver function using lru_cache for memoization.
        # @lru_cache stores the results of function calls based on their arguments.
        # If `solve` is called again with the same `current_nums_tuple`, the cached
        # result is returned instantly, avoiding redundant computations.
        # `maxsize=None` allows the cache to grow without a size limit, which is
        # suitable here as the number of distinct subproblem states might be large but finite.
        @lru_cache(maxsize=None)
        def solve(current_nums_tuple: Tuple[int, ...]) -> int:
            """
            Recursively computes the minimum cost to remove elements from the current tuple state.
            
            Args:
                current_nums_tuple: The tuple representing the remaining elements in the current subproblem.
            
            Returns:
                The minimum cost to empty this tuple starting from the current state.
            """
            # Get the number of elements remaining in the current state.
            m = len(current_nums_tuple)

            # Base Cases: These handle the termination conditions of the recursion.
            # According to the problem statement, if fewer than three elements remain,
            # they are removed in a single final operation.
            if m == 0:
                # If the tuple is empty, no elements remain, so the cost is 0.
                return 0
            if m == 1:
                # If one element remains, the cost of removing it is its value.
                return current_nums_tuple[0]
            if m == 2:
                # If two elements remain, the cost of removing them is the maximum of the two values.
                return max(current_nums_tuple[0], current_nums_tuple[1])

            # Recursive Step (m >= 3):
            # When three or more elements remain, we must choose one of the three allowed operations.
            # An operation involves selecting two elements from the first three and removing them.
            
            # Extract the first three elements for easier reference.
            nums0 = current_nums_tuple[0]
            nums1 = current_nums_tuple[1]
            nums2 = current_nums_tuple[2]

            # Explore the three possible operations and calculate their costs recursively:

            # Operation 1: Remove the first two elements (at indices 0 and 1).
            # The cost of this operation is max(nums0, nums1).
            # The remaining elements form a suffix starting from index 2.
            # The total cost for this choice is the operation cost plus the minimum cost
            # to remove the remaining elements (solved recursively).
            # Tuple slicing `current_nums_tuple[2:]` creates the tuple for the subproblem.
            remaining1 = current_nums_tuple[2:] 
            cost1 = max(nums0, nums1) + solve(remaining1)

            # Operation 2: Remove the first and third elements (at indices 0 and 2).
            # The cost of this operation is max(nums0, nums2).
            # The remaining elements consist of the second element (nums1) followed by
            # the suffix of the original tuple starting from index 3.
            # Tuple concatenation `(nums1,) + current_nums_tuple[3:]` forms the tuple for the subproblem.
            remaining2 = (nums1,) + current_nums_tuple[3:]
            cost2 = max(nums0, nums2) + solve(remaining2)

            # Operation 3: Remove the second and third elements (at indices 1 and 2).
            # The cost of this operation is max(nums1, nums2).
            # The remaining elements consist of the first element (nums0) followed by
            # the suffix of the original tuple starting from index 3.
            remaining3 = (nums0,) + current_nums_tuple[3:]
            cost3 = max(nums1, nums2) + solve(remaining3)

            # The minimum cost for the current state `current_nums_tuple` is the minimum
            # of the total costs resulting from each of the three possible operations.
            return min(cost1, cost2, cost3)

        # Start the computation by calling the recursive solver function with the
        # initial tuple representing the entire input array `nums`.
        return solve(nums_tuple)