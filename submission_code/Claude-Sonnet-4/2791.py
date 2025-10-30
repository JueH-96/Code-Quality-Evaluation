class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Track which friends have received the ball
        received = set()
        
        # Start at friend 1 (index 0)
        current = 0
        turn = 1
        
        while current not in received:
            # Mark current friend as having received the ball
            received.add(current)
            
            # Calculate next position: move turn * k steps clockwise
            current = (current + turn * k) % n
            turn += 1
        
        # Find friends who never received the ball
        losers = []
        for i in range(n):
            if i not in received:
                losers.append(i + 1)  # Convert back to 1-based indexing
        
        return losers