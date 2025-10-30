class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        q = list(range(n))
        wins = [0] * n
        
        while True:
            p1_idx = q.pop(0)
            p2_idx = q.pop(0)
            
            if skills[p1_idx] > skills[p2_idx]:
                winner_idx = p1_idx
                loser_idx = p2_idx
            else:
                winner_idx = p2_idx
                loser_idx = p1_idx
                
            wins[winner_idx] += 1
            
            if wins[winner_idx] == k:
                return winner_idx
            
            q.insert(0, winner_idx)
            q.append(loser_idx)