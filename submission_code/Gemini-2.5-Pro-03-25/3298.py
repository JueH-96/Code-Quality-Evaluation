import collections
from typing import List

class Solution:
    """
    Solves the problem of finding the maximum number of elements that can be selected
    from a modified array (each element increased by at most 1) such that the selected
    elements form a consecutive sequence when sorted.
    Uses dynamic programming on the sorted array.
    """
    def maxSelectedElements(self, nums: List[int]) -> int:
        """
        Calculates the maximum number of selected elements forming a consecutive sequence.

        Args:
            nums: A 0-indexed array of positive integers.

        Returns:
            The maximum number of elements that can be selected.
        """

        # Sort the input array to process elements in increasing order.
        # This allows building consecutive sequences incrementally based on previous results.
        # Time complexity for sorting is O(N log N).
        nums.sort()
        
        # dp[v] stores the maximum length of a consecutive sequence ending exactly at value v,
        # considering the elements processed so far.
        # Using collections.defaultdict(int) simplifies code by providing a default value of 0 
        # for keys that haven't been assigned a value yet. This effectively handles the base cases
        # where a sequence starts.
        # Space complexity: O(N) in the worst case, as there can be up to 2N distinct values 
        # (x and x+1 for each element x in nums) stored as keys in the dictionary.
        dp = collections.defaultdict(int)
        
        # Initialize the overall maximum length found. Since the problem constraints state 
        # nums.length >= 1, there will always be at least one element. A single element 
        # can form a sequence of length 1. Therefore, max_len will be at least 1 if nums is non-empty.
        # Initializing to 0 is fine as it will be updated correctly.
        max_len = 0
        
        # Iterate through each number x in the sorted array.
        # The loop runs N times, where N is the length of nums.
        # Inside the loop, dictionary operations (get, set) take average O(1) time.
        # Total time complexity for the loop is O(N).
        for x in nums:
            # Calculate the potential length if we use the current element x transformed into x+1.
            # This potentially extends a consecutive sequence ending at x. The length would be dp[x] + 1.
            # We read dp[x] which represents the max length ending at x using elements processed *before* the current x.
            len_x_plus_1 = dp[x] + 1
            
            # Calculate the potential length if we use the current element x as its original value x.
            # This potentially extends a consecutive sequence ending at x-1. The length would be dp[x-1] + 1.
            len_x = dp[x - 1] + 1
            
            # Update the DP table. The order of updates is crucial to ensure correctness.
            # We must update dp[x+1] *before* updating dp[x]. 
            # This ensures that the calculation for len_x_plus_1 (which depends on dp[x])
            # uses the value of dp[x] derived from elements processed *before* the current element x potentially modifies dp[x].
            # If we updated dp[x] first, then len_x_plus_1 might incorrectly use a length
            # that already incorporated the current element x (when used as x), effectively using the same element twice.
            
            # Update the maximum length for a sequence ending at x+1.
            # We take the maximum of its current value (if any existed) and the newly calculated potential length.
            dp[x + 1] = max(dp[x + 1], len_x_plus_1)
            
            # Update the maximum length for a sequence ending at x.
            # Similarly, take the maximum of its current value and the newly calculated potential length.
            dp[x] = max(dp[x], len_x)
            
            # Update the overall maximum length found so far.
            # After processing element x, the longest sequence might end at value x or value x+1.
            # We compare the current max_len with the lengths stored in dp[x] and dp[x+1] after their updates.
            # This ensures max_len always holds the maximum length achieved up to the current element.
            max_len = max(max_len, dp[x], dp[x + 1])

        # The variable max_len tracks the maximum length found across all steps and all possible sequence ending values.
        # Since nums is guaranteed non-empty by constraints (1 <= nums.length), max_len will be at least 1.
        # Return the final maximum length.
        # Overall time complexity: O(N log N) due to sorting.
        # Overall space complexity: O(N) for the DP dictionary.
        return max_len