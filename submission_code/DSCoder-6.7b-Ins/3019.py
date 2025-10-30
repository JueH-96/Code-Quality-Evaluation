class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left = right = 0
        for move in moves:
            if move == 'L':
                left += 1
            elif move == 'R':
                right += 1
            else:
                if left > right:
                    left += 1
                else:
                    right += 1
        return max(left, right)