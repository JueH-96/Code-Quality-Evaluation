class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        l_count = 0
        r_count = 0
        underscore_count = 0
        for move in moves:
            if move == 'L':
                l_count += 1
            elif move == 'R':
                r_count += 1
            elif move == '_':
                underscore_count += 1
        
        right_pos = r_count + underscore_count - l_count
        left_pos = r_count - (l_count + underscore_count)
        
        return max(abs(right_pos), abs(left_pos))