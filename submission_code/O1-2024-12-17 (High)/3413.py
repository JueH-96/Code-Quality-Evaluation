class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        from collections import deque
        
        n = len(skills)
        # Find the index of the player with the maximum (unique) skill
        max_index = 0
        for i in range(1, n):
            if skills[i] > skills[max_index]:
                max_index = i
        
        # We'll simulate the competition using a queue for challengers.
        # The front of the queue is the next challenger; we'll keep track
        # of the current champion and how many consecutive wins they have.
        champion = 0
        consecutive_wins = 0
        queue = deque(range(1, n))
        
        # Keep matching until someone reaches k wins or the champion is unstoppable (max skill).
        while True:
            challenger = queue[0]
            
            if skills[champion] > skills[challenger]:
                # Champion wins again
                consecutive_wins += 1
                # The loser (challenger) goes to the back of the queue
                queue.append(queue.popleft())
            else:
                # Challenger dethrones the champion
                consecutive_wins = 1
                old_champion = champion
                champion = challenger
                queue.popleft()
                queue.append(old_champion)
            
            # If current champion has k consecutive wins, return immediately
            if consecutive_wins >= k:
                return champion
            
            # If the champion is the player with the max skill, no one can ever beat them,
            # so eventually they will get k consecutive wins. We can return immediately.
            if champion == max_index:
                return champion