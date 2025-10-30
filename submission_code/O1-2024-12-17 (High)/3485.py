class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort the starting points
        start.sort()
        n = len(start)
        
        # Binary search boundaries for the possible answer
        # The maximum possible difference cannot exceed (max(start)+d) - min(start).
        left, right = 0, (start[-1] + d) - start[0]
        
        # Check feasibility of a given minimum gap "gap"
        def can_place(gap: int) -> bool:
            # Greedily place the first chosen integer at start[0]
            last_chosen = start[0]
            
            # Try placing each subsequent integer in sorted order
            for i in range(1, n):
                # We need to place the next integer at least "gap" away from last_chosen
                candidate = max(start[i], last_chosen + gap)
                # If candidate is out of the i-th interval [start[i], start[i] + d], fail
                if candidate > start[i] + d:
                    return False
                # Otherwise, accept candidate
                last_chosen = candidate
            
            return True
        
        # Binary search for the maximum feasible gap
        while left < right:
            mid = (left + right + 1) // 2
            if can_place(mid):
                left = mid
            else:
                right = mid - 1
        
        return left