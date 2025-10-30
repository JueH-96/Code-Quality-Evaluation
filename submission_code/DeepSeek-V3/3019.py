class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Initialize counters for L, R, and _
        l_count = 0
        r_count = 0
        underscore_count = 0
        
        for move in moves:
            if move == 'L':
                l_count += 1
            elif move == 'R':
                r_count += 1
            else:
                underscore_count += 1
        
        # The maximum distance is the absolute difference between the total L and R moves
        # plus the number of underscores, all directed to the side with more L or R
        if l_count > r_count:
            return abs(l_count - r_count + underscore_count)
        else:
            return abs(r_count - l_count + underscore_count)