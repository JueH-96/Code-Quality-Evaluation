from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)

        def can_achieve(target_X: int) -> bool:
            if target_X == 0:
                return True

            # Calculate required visits v_i for each point i
            v = [(target_X + points[i] - 1) // points[i] if points[i] > 0 else float('inf') for i in range(n)]
            # If target_X > 0 and points[i] is 0, it's impossible if v[i] becomes non-zero.
            # However, constraints say points[i] >= 1, so points[i] > 0 always.
            
            # Determine R_max, the rightmost point that needs visiting
            R_max = -1
            for i in range(n - 1, -1, -1):
                if v[i] > 0:
                    R_max = i
                    break
            
            if R_max == -1: # All v[i] are 0 (e.g. target_X is small)
                return True # Cost is 0 moves.

            # Option 1: Path -1 -> 0 -> ... -> R_max (ends at R_max)
            # Base moves for path 0 -> R_max is R_max. Add 1 for initial -1 -> 0. Total 1 + R_max.
            # Base visits: N_i = 1 for i in [0, R_max].
            cost1 = 1 + R_max
            for i in range(R_max + 1):
                cost1 += max(0, v[i] - 1) * 2
            
            # If R_max is 0, path is just -1 -> 0. Only this option is meaningful.
            if R_max == 0:
                return cost1 <= m

            # Option 2: Path -1 -> 0 -> ... -> R_max -> ... -> 0 (ends at 0)
            # Base moves for path 0 -> R_max -> 0 is 2*R_max. Add 1 for initial -1 -> 0. Total 1 + 2*R_max.
            # Base visits: N_0=2. N_i=2 for i in (0, R_max). N_R_max=1.
            cost2 = 1 + 2 * R_max
            
            # Contribution from v[0]
            cost2 += max(0, v[0] - 2) * 2
            # Contribution from v[i] for 0 < i < R_max
            for i in range(1, R_max):
                cost2 += max(0, v[i] - 2) * 2
            # Contribution from v[R_max]
            cost2 += max(0, v[R_max] - 1) * 2
                
            return min(cost1, cost2) <= m

        # Binary search for the maximum X
        low = 0
        # Max possible points[i] is 10^6, m is 10^9. Max score could be 10^15.
        high = m * max(points) + 1 # A sufficiently large upper bound for X
        ans = 0

        while low <= high:
            mid = low + (high - low) // 2
            if mid < 0: # Should not happen with non-negative low/high
                mid = 0 
            
            if can_achieve(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans