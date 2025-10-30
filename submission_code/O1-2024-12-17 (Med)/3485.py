class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort intervals by their starting points
        intervals = sorted((s, s + d) for s in start)
        
        # Greedy check to see if we can place all n integers
        # with at least "gap" distance between any two chosen integers
        def can_place(gap: int) -> bool:
            # Place the first integer at the start of the first interval
            chosen = intervals[0][0]
            
            # Try placing remaining integers
            for i in range(1, len(intervals)):
                left, right = intervals[i]
                # We need to place an integer >= (chosen + gap)
                # but also within [left, right].
                # So we pick next_position = max(chosen + gap, left).
                if chosen + gap > right:
                    return False  # cannot place in this interval
                chosen = max(chosen + gap, left)
            
            return True
        
        # Binary search for the maximum possible gap
        min_start = min(start)
        max_start = max(start)
        
        left, right = 0, (max_start + d) - min_start
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if can_place(mid):
                result = mid  # mid is feasible, try for a bigger gap
                left = mid + 1
            else:
                right = mid - 1
        
        return result