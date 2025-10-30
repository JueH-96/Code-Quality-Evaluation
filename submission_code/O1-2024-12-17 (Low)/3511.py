class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # The only way to sum 115 with coins of 75 and 10
        # is exactly 1 coin of 75 and 4 coins of 10 each turn.
        # The maximum number of full turns possible is:
        # t = min(x, y // 4).
        #
        # Turns are taken: Alice (1st), Bob (2nd), Alice (3rd), etc.
        # If t is odd, the t-th turn belongs to Alice, so Bob cannot move next and loses.
        # If t is even, the t-th turn belongs to Bob, so Alice cannot move next and loses.
        #
        # We return the name of the winning player.
        
        t = min(x, y // 4)
        if t % 2 == 1:
            # t is odd -> last turn is Alice's -> Bob loses -> Alice wins
            return "Alice"
        else:
            # t is even -> last turn is Bob's -> Alice loses -> Bob wins
            return "Bob"