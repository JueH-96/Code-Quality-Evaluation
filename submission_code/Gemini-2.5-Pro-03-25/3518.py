import math
from typing import List

class Solution:
    """
    Finds the maximum score achievable by selecting 4 indices i_0 < i_1 < i_2 < i_3 from array b,
    where the score is a[0]*b[i_0] + a[1]*b[i_1] + a[2]*b[i_2] + a[3]*b[i_3].
    Uses dynamic programming with O(n) time and O(n) space complexity.
    """
    def maxScore(self, a: List[int], b: List[int]) -> int:
        """
        Calculates the maximum score using dynamic programming.

        Args:
            a: A list of 4 integers (coefficients).
            b: A list of at least 4 integers.

        Returns:
            The maximum achievable score as an integer.
        """
        n = len(b)
        # Use negative infinity for initialization to correctly handle negative scores.
        # Python's arbitrary precision integers handle large values, but float('-inf') 
        # provides a convenient way to represent the lowest possible value.
        neg_inf = -math.inf

        # Let dp[k][i] be the maximum score using the first k elements of 'a' (a[0]...a[k-1])
        # by choosing k elements from 'b' with indices j_0 < j_1 < ... < j_{k-1} <= i.
        
        # We use two arrays to optimize space from O(k*n) to O(n).
        # prev_dp stores dp values for state k-1.
        # curr_dp stores dp values for state k.
        prev_dp = [neg_inf] * n
        
        # Base case: k = 1 (using a[0])
        # dp[1][i] = max score using a[0] and ONE element from b[0...i].
        # This means dp[1][i] = max_{0 <= j <= i} (a[0] * b[j]).
        
        max_val_so_far = neg_inf # Tracks the maximum value of a[0]*b[j] seen so far up to index i.
        for i in range(n):
            val = a[0] * b[i]
            max_val_so_far = max(max_val_so_far, val)
            prev_dp[i] = max_val_so_far # Store dp[1][i]

        # Iterate for k = 2, 3, 4
        # The outer loop variable 'k' represents the number of elements from 'a' used (1-based index).
        # Corresponds to DP state k. We compute states k=2, 3, 4.
        for k in range(2, 5): # k iterates through 2, 3, 4
            curr_dp = [neg_inf] * n # Initialize dp array for the current state k
            
            # current_max_score_k tracks the value of dp[k][i] as we iterate through i.
            # It represents the maximum score using k elements ending at or before index i.
            current_max_score_k = neg_inf 

            # The recurrence relation is:
            # dp[k][i] = max(dp[k][i-1], f[k][i])
            # where f[k][i] is the maximum score using k elements with the k-th element being b[i].
            # f[k][i] = dp[k-1][i-1] + a[k-1] * b[i]
            
            # The k-th element b[j_{k-1}] must be at an index i >= k-1 
            # because we need k-1 elements before it at smaller indices.
            for i in range(k - 1, n):
                # Calculate f[k][i]: score if the k-th element chosen is b[i].
                # This requires the max score using k-1 elements ending at or before index i-1, which is dp[k-1][i-1].
                # dp[k-1][i-1] is stored in prev_dp[i-1].
                
                f_k_i = neg_inf # Initialize score for this path to negative infinity
                if prev_dp[i-1] != neg_inf:
                    # Only proceed if a valid sequence of k-1 elements ending at or before i-1 exists.
                    f_k_i = prev_dp[i-1] + a[k-1] * b[i]
                
                # Calculate dp[k][i] = max(dp[k][i-1], f[k][i])
                # dp[k][i-1] is the value of 'current_max_score_k' from the previous iteration of i.
                if i == k - 1:
                    # Base case for this k: dp[k][k-1] = f[k][k-1], as dp[k][k-2] is effectively -infinity.
                    current_max_score_k = f_k_i
                else:
                    # Take the maximum of either ending the sequence at index i (f[k][i])
                    # or the maximum score found ending before index i (dp[k][i-1]).
                    current_max_score_k = max(current_max_score_k, f_k_i)
                
                # Store the computed dp[k][i] value.
                curr_dp[i] = current_max_score_k

            # Update prev_dp to curr_dp for the next iteration of k
            prev_dp = curr_dp

        # After the loops complete, prev_dp holds the values for k=4.
        # The final result is dp[4][n-1], which is the maximum score using 4 elements from 'a'
        # and choosing 4 elements from 'b' with indices up to n-1.
        result = prev_dp[n-1]
        
        # The problem guarantees n >= 4, implying that a valid selection of 4 indices always exists.
        # Thus, the maximum score should be a finite number (possibly negative).
        # Cast the final result to int as required by the return type.
        # Python's float('-inf') can cause issues if directly cast to int.
        # However, since a solution must exist, result should not remain -inf.
        return int(result)