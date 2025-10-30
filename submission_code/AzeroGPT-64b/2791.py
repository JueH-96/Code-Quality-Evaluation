class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set()  # To keep track of the friends who have received the ball
        friend = 0  # Index of the current friend
        turn = 1  # Turn counter
        
        while friend not in received:
            received.add(friend)
            friend = (friend + turn * k) % n  # Calculate the index of the next friend
            turn += 1
        
        # Find the losers: friends who have not received the ball
        losers = [i + 1 for i in range(n) if i not in received]
        
        return losers