class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Set to track friends who have received the ball
        received = set()
        
        # Current position (0-indexed, so friend 1 is at position 0)
        current = 0
        
        # Turn counter
        turn = 1
        
        # Continue until someone receives the ball twice
        while current not in received:
            received.add(current)
            # Pass the ball turn*k steps clockwise
            current = (current + turn * k) % n
            turn += 1
        
        # Find losers (friends who never received the ball)
        losers = []
        for i in range(n):
            if i not in received:
                # Convert back to 1-indexed
                losers.append(i + 1)
        
        return losers