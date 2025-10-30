class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each turn requires 1 coin of 75 and 4 coins of 10
        # Maximum turns possible
        turns = min(x, y // 4)
        
        # If odd number of turns, Alice wins (she plays the last turn)
        # If even number of turns, Bob wins (he plays the last turn)
        return "Alice" if turns % 2 == 1 else "Bob"