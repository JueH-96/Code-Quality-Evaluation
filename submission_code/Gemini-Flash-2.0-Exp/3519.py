class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        wins = 0
        for i in range(n):
            counts = {}
            for p in pick:
                if p[0] == i:
                    if p[1] not in counts:
                        counts[p[1]] = 0
                    counts[p[1]] += 1
            
            won = False
            for color in counts:
                if counts[color] > i:
                    won = True
                    break
            
            if won:
                wins += 1
        
        return wins