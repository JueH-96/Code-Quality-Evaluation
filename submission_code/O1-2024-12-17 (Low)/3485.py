class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort the starting points
        start.sort()
        
        # Helper function to check feasibility of a given distance "mid"
        # We'll greedily place each chosen number, ensuring each is
        # at least "mid" away from the previous chosen number and
        # also within its [start[i], start[i] + d] range.
        def can_place(mid):
            # Place the first chosen number at start[0]
            chosen = start[0]
            
            for i in range(1, len(start)):
                # We want to place the next chosen number at least "mid" away from "chosen"
                # but it must also be within [start[i], start[i]+d].
                # So we try next_pos = max(start[i], chosen + mid).
                next_pos = max(start[i], chosen + mid)
                if next_pos > start[i] + d:
                    # Not possible to place the point in the i-th interval
                    # while maintaining distance "mid".
                    return False
                # Update chosen to this next position
                chosen = next_pos
            return True
        
        # We'll binary search on the answer (the minimum distance).
        # The maximum possible distance is at most (max(start) + d) - min(start).
        left, right = 0, (start[-1] + d) - start[0]
        ans = 0
        
        while left <= right:
            mid = (left + right) // 2
            if can_place(mid):
                ans = mid  # mid is feasible, try for a larger distance
                left = mid + 1
            else:
                right = mid - 1
        
        return ans