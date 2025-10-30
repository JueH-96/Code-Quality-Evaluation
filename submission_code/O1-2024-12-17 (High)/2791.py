class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        from typing import List
        
        received = set([1])  # Friend 1 starts with the ball
        current = 1          # Current holder of the ball
        step = 1             # i-th turn, multiplied by k
        
        while True:
            # Calculate the next friend (1-indexed)
            next_friend = (current - 1 + step * k) % n + 1
            
            # If this friend has already received the ball, game ends
            if next_friend in received:
                break
            
            # Otherwise, mark the friend as having received the ball
            received.add(next_friend)
            # Update current friend
            current = next_friend
            
            # Move to the next step
            step += 1
        
        # Losers are friends who never received the ball
        losers = [i for i in range(1, n+1) if i not in received]
        return losers