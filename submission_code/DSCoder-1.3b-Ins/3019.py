class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        x, y = 0, 0
        max_distance = 0
        for move in moves:
            if move == 'R':
                x += 1
            elif move == 'L':
                x -= 1
            elif move == '_':
                y += 1
            max_distance = max(max_distance, abs(x) + abs(y))
        return max_distance