class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        
        while True:
            if len(queue) < k:
                return queue[0]
            
            winner_count = 0
            winner = queue[0]
            
            for i in range(k):
                player1 = queue[i]
                player2 = queue[i + 1]
                
                if skills[player1] > skills[player2]:
                    queue.append(queue.pop(i + 1))
                    winner_count += 1
                else:
                    queue.append(queue.pop(i))
                    break
            
            if winner_count == k:
                return winner