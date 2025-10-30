class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        r_count = 0
        l_count = 0
        underscore = 0
        for move in moves:
            if move == 'R':
                r_count += 1
            elif move == 'L':
                l_count += 1
            else:
                underscore += 1
        base = r_count - l_count
        return max(abs(base + underscore), abs(base - underscore))