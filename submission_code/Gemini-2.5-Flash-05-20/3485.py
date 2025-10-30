from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        
        # Create intervals as (start_point, end_point) tuples.
        # end_point is start_point + d.
        intervals = []
        for i in range(n):
            intervals.append((start[i], start[i] + d))
        
        # Sort intervals by their end points.
        # This is a crucial step for the greedy strategy in the check function.
        # If end points are identical, the order of intervals by their start points
        # does not affect correctness, as the choice for next_val handles `s` correctly.
        intervals.sort(key=lambda x: x[1])

        def check(min_diff: int) -> bool:
            """
            Checks if it's possible to choose n integers, one from each interval,
            such that the minimum absolute difference between any two chosen integers
            is at least 'min_diff'.
            
            The greedy strategy here is to iterate through intervals sorted by their end points.
            For each interval, choose the smallest possible integer that satisfies
            the 'min_diff' requirement with respect to the previously chosen integer.
            """
            
            # current_chosen_val stores the value picked for the interval that was processed previously
            # in the `intervals` (sorted by end_point) list.
            # Initialize with a very small number, effectively negative infinity,
            # so that the first chosen value can be its interval's start point.
            current_chosen_val = -float('inf') 
            
            for s, e in intervals:
                # To satisfy the `min_diff` requirement, the next chosen value must be
                # at least `current_chosen_val + min_diff`.
                # Also, it must be within the current interval [s, e], meaning it must be >= s.
                # So, the smallest valid value we can pick for the current interval is:
                next_val = max(s, current_chosen_val + min_diff)
                
                # If this smallest valid `next_val` exceeds the end point `e` of the current interval,
                # it means we cannot pick any integer from this interval that satisfies all conditions.
                # Thus, `min_diff` is not achievable.
                if next_val > e:
                    return False
                
                # If a valid `next_val` is found, we choose this smallest value.
                # This greedy choice leaves maximum room for subsequent choices,
                # increasing the chance of overall success.
                current_chosen_val = next_val
            
            # If we successfully picked a value for all intervals, `min_diff` is achievable.
            return True
        
        # Binary search for the maximum possible 'min_diff'.
        # The range of possible 'min_diff' values:
        # 'low' can be 0 (if all intervals overlap greatly).
        # 'high' can be the maximum possible difference between any start[i]+d and start[j].
        # Maximum start[i] + d = 10^9 + 10^9 = 2 * 10^9.
        # Minimum start[i] = 0.
        # So, the largest possible difference between two chosen points is roughly 2 * 10^9.
        # Thus, 'min_diff' cannot exceed this. A slightly larger upper bound is safe.
        low = 0
        high = 2 * 10**9 + 7 # A sufficiently large upper bound.
        ans = 0 # Stores the maximum 'min_diff' found so far that is achievable.
        
        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                # If 'mid' is an achievable minimum difference, it means we might be able to do even better.
                # Store 'mid' as a potential answer and try to achieve a larger 'min_diff'.
                ans = mid
                low = mid + 1
            else:
                # If 'mid' is not achievable, we need to try a smaller minimum difference.
                high = mid - 1
                
        return ans