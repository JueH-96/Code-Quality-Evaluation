class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        min_pos = max_pos = 0
        for move in moves:
            if move == 'L':
                min_pos -= 1
                max_pos -= 1
            elif move == 'R':
                min_pos += 1
                max_pos += 1
            else:
                min_pos -= 1
                max_pos += 1
        return max(abs(min_pos), abs(max_pos))