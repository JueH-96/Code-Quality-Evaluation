class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each turn, a player must pick up coins with a total value 115.
        # This can be achieved by picking 1 coin with a value of 75 and 4 coins with a value of 10.
        # Therefore, the total number of turns that can be played is the minimum of the number of 75 coins and the number of 10 coins divided by 4.
        turns = min(x, y // 4)
        
        # If the number of turns is even, Alice loses because Bob plays the last turn.
        # If the number of turns is odd, Bob loses because Alice plays the last turn.
        return "Alice" if turns % 2 == 0 else "Bob"