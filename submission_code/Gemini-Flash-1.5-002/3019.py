class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        l_count = moves.count('L')
        r_count = moves.count('R')
        blank_count = moves.count('_')
        
        max_dist = abs(r_count + blank_count - l_count)
        min_dist = abs(r_count - (l_count + blank_count))

        return max(max_dist,min_dist)