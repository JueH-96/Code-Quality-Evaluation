class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        from collections import deque
        
        n = len(skills)
        # Special quick check: if k == 1, then the winner is simply
        # whoever wins the very first match (the maximum of the first two).
        # But the general logic below handles k==1 fine as well, so no special case needed.

        # Find index of the player with the maximum skill (cannot be dethroned once on top).
        max_index = 0
        for i in range(1, n):
            if skills[i] > skills[max_index]:
                max_index = i
        
        # Initialize a queue of player indices: [0, 1, 2, ..., n-1]
        queue = deque(range(n))
        
        # The front of the queue will be the current champion.
        champ = queue.popleft()
        consecutive = 0  # Number of consecutive wins by the current champion.
        
        while True:
            challenger = queue.popleft()
            
            if skills[champ] > skills[challenger]:
                # Champion wins again
                consecutive += 1
                # The loser goes to the back of the queue
                queue.append(challenger)
            else:
                # Challenger dethrones the champion
                queue.append(champ)
                champ = challenger
                consecutive = 1
            
            # If the champion just reached k consecutive wins, return immediately
            if consecutive >= k:
                return champ
            
            # If the champion is the player with the maximum skill, no one can dethrone them.
            # They will eventually reach k consecutive wins, so we can return now.
            if champ == max_index:
                return champ