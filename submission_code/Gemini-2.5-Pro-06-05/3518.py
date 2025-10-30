from typing import List
import math

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        """
        Calculates the maximum score by choosing 4 elements from b at increasing indices.

        The method uses dynamic programming. Let dp[k][j] be the maximum score for
        choosing k elements, with the k-th element being b[j]. The recurrence is:
        dp[k][j] = a[k-1]*b[j] + max_{l < j}(dp[k-1][l]).

        This is implemented with O(n) space by only keeping track of the previous
        and current DP states.
        """
        n = len(b)
        
        # dp[j] will store the max score for choosing `k` elements,
        # with the k-th element being b[j].
        
        # Base case: k=1 (using a[0])
        # The score is simply a[0] * b[j].
        dp = [a[0] * val for val in b]
        
        # Iteratively compute for k=2, 3, 4 (using a[1], a[2], a[3]).
        # The outer loop index `i` corresponds to the index in `a`.
        for i in range(1, 4):
            # Store the dp values from the previous iteration (k-1)
            prev_dp = dp
            # Initialize dp for the current iteration (k)
            dp = [-math.inf] * n
            
            # max_so_far will track max(prev_dp[l]) for l < j
            max_so_far = -math.inf
            
            # To choose i+1 elements, the last one (b[j]) must have index j >= i.
            # e.g., for 2 elements, the second one's index must be at least 1.
            for j in range(i, n):
                # Update the running maximum of scores from the previous stage
                max_so_far = max(max_so_far, prev_dp[j-1])
                
                # Calculate the DP value for the current stage
                dp[j] = max_so_far + a[i] * b[j]
        
        # The final result is the maximum score after choosing all 4 elements.
        # This is the maximum value in the final dp array, as the 4th element can
        # be any b[j] for j >= 3.
        return int(max(dp))