class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        current_min = current_max = 0
        for move in moves:
            if move == 'L':
                current_min -= 1
                current_max -= 1
            elif move == 'R':
                current_min += 1
                current_max += 1
            else:  # '_'
                current_min -= 1
                current_max += 1
        return max(abs(current_min), abs(current_max))