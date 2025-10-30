class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        consecutive_wins = 0
        last_winner = -1
        
        while consecutive_wins < k:
            player1 = queue[0]
            player2 = queue[1]
            
            # Compare skills and determine winner
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            # Update queue - winner stays at front, loser goes to back
            queue.pop(0)
            queue.pop(0)
            queue.insert(0, winner)
            queue.append(loser)
            
            # Update consecutive wins count
            if winner == last_winner:
                consecutive_wins += 1
            else:
                consecutive_wins = 1
                last_winner = winner
                
        return last_winner