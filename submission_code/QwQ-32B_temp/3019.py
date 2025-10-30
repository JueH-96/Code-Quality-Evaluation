class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        underscores = len(moves) - count_L - count_R
        
        scenario1 = (count_R + underscores) - count_L
        scenario2 = count_R - (count_L + underscores)
        
        return max(abs(scenario1), abs(scenario2))