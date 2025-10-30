class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize a set to store the losers
        losers = set()
        
        # Initialize the current player to 1
        current_player = 1
        
        # Keep track of the players who have received the ball
        received_ball = set([current_player])
        
        # Play the game until a player receives the ball for the second time
        while True:
            # Calculate the next player to receive the ball
            next_player = (current_player + k) % n
            if next_player == 0:
                next_player = n
            
            # If the next player has already received the ball, the game is over
            if next_player in received_ball:
                break
            
            # Add the current player to the set of players who have received the ball
            received_ball.add(next_player)
            
            # Move to the next player
            current_player = next_player
        
        # Add all the players who have not received the ball to the set of losers
        for i in range(1, n+1):
            if i not in received_ball:
                losers.add(i)
        
        # Return the losers in ascending order
        return sorted(list(losers))