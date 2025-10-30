class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = 0
        right_count = 0
        underscore_count = 0
        for move in moves:
            if move == 'L':
                left_count += 1
            elif move == 'R':
                right_count += 1
            else:
                underscore_count += 1
        
        return abs(right_count + underscore_count - left_count) if right_count + underscore_count > left_count else abs(left_count + underscore_count - right_count)