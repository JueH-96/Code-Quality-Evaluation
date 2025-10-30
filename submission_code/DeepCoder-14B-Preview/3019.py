class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_R = moves.count('R')
        count_L = moves.count('L')
        count_U = moves.count('_')
        
        D = count_R - count_L - count_U
        option1 = D
        option2 = D + 2 * count_U
        
        return max(abs(option1), abs(option2))