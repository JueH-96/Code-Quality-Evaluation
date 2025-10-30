class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize the set of friends who have received the ball
        received = set()
        # Start with the first friend
        current = 1
        # Initialize the turn counter
        turn = 1
        
        # Loop until a friend receives the ball for the second time
        while current not in received:
            # Mark the current friend as having received the ball
            received.add(current)
            # Calculate the next friend to receive the ball
            current = (current + turn * k - 1) % n + 1
            # Increment the turn counter
            turn += 1
        
        # Find the friends who did not receive the ball
        losers = [i for i in range(1, n + 1) if i not in received]
        
        return losers