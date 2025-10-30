class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort intervals by their start points
        start.sort()
        
        # Helper function to check feasibility of a candidate difference "mid"
        def can_place(mid: int) -> bool:
            # Place the first chosen number at the leftmost point in the first interval
            chosen = start[0]
            
            # Go through each subsequent interval and try placing
            for i in range(1, len(start)):
                # The next chosen must be at least chosen + mid
                needed = chosen + mid
                # The i-th interval is [start[i], start[i] + d]
                # If needed <= start[i] + d, we can place it
                if needed > start[i] + d:
                    return False
                # Place as far left as possible while >= needed
                # That is max(start[i], needed)
                chosen = max(start[i], needed)
            return True
        
        # Binary search boundaries:
        # The maximum min-difference cannot exceed the span of (max(start)+d - min(start))
        left, right = 0, (max(start) + d) - min(start)
        
        # Binary search for the maximum feasible difference
        while left < right:
            mid = (left + right + 1) // 2
            if can_place(mid):
                left = mid  # mid is feasible, try a bigger one
            else:
                right = mid - 1
        
        return left