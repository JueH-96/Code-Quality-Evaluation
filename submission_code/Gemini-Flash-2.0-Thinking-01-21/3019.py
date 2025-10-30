class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = 0
        count_R = 0
        count_underscore = 0
        for move in moves:
            if move == 'L':
                count_L += 1
            elif move == 'R':
                count_R += 1
            elif move == '_':
                count_underscore += 1
        
        distance_left = abs(count_R - count_L - count_underscore)
        distance_right = abs(count_R + count_underscore - count_L)
        
        return max(distance_left, distance_right)