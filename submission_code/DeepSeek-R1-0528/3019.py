class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        free = len(moves) - left_count - right_count
        option1 = right_count - left_count + free
        option2 = right_count - left_count - free
        return max(abs(option1), abs(option2))