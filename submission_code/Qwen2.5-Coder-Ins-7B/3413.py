class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        player_indices = list(range(n))
        wins = [0] * n
        
        for _ in range(n):
            i, j = player_indices[:2]
            if skills[i] > skills[j]:
                player_indices = [i] + player_indices[2:] + [j]
                wins[i] += 1
                wins[j] = 0
                if wins[i] == k:
                    return i
            else:
                player_indices = [j] + player_indices[2:] + [i]
                wins[j] += 1
                wins[i] = 0
                if wins[j] == k:
                    return j