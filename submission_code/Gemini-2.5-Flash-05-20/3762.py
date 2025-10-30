import math
from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)

        # Binary search range for the maximum possible minimum score (X).
        # low: The smallest possible minimum score (0, achievable with 0 moves).
        # high: A sufficiently large upper bound.
        # Max points[i] is 10^6, max m is 10^9.
        # If all 'm' moves are used to visit a single index 'i', gameScore[i] would be m * points[i].
        # So, 10^9 * 10^6 = 10^15 is a safe upper bound for any individual gameScore[i] value,
        # and thus for the minimum score X.
        low = 0
        high = 10**15 # Max m * Max points[i] = 10^9 * 10^6 = 10^15
        
        ans = 0 # Stores the maximum X found so far that is achievable.

        # Helper function: Checks if a minimum score of `X` is achievable within `m` moves.
        def check(X: int) -> bool:
            # If X is 0, no visits are required for any index, so 0 moves are needed.
            # This is always possible (by doing nothing).
            if X == 0:
                return True 

            total_moves_needed = 0
            for p_val in points:
                # Calculate the minimum number of visits needed for this index 'i'
                # to ensure gameScore[i] >= X.
                # visits[i] = ceil(X / points[i]).
                # Using integer division: (numerator + divisor - 1) // divisor
                visits = (X + p_val - 1) // p_val
                
                total_moves_needed += visits
                
                # Optimization: If total_moves_needed already exceeds 'm',
                # we can stop early as X is not achievable.
                if total_moves_needed > m:
                    return False
            
            # If the total moves needed is less than or equal to 'm', then X is achievable.
            return True # total_moves_needed <= m is implicitly true here

        # Perform binary search
        while low <= high:
            mid = low + (high - low) // 2
            
            if check(mid):
                # If 'mid' is achievable, it's a potential answer.
                # Try to find a larger 'X' by searching in the upper half.
                ans = mid
                low = mid + 1 
            else:
                # If 'mid' is not achievable, it's too high.
                # Search in the lower half for a smaller 'X'.
                high = mid - 1 

        return ans