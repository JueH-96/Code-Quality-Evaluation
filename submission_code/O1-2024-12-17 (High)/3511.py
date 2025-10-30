class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each valid move requires picking exactly:
        # 1 coin of value 75 and 4 coins of value 10,
        # because 75 + (4*10) = 115 and there is no other way
        # to form 115 using 75 and 10.
        
        # The maximum number of moves (turns) is limited by how many times
        # we can remove (1 coin of 75, 4 coins of 10):
        T = min(x, y // 4)
        
        # Players alternate turns: Alice picks on odd turns, Bob picks on even turns.
        # If the total count of moves T is odd, Alice is the last to pick and wins.
        # If T is even, Bob is the last to pick and wins.
        return "Alice" if T % 2 == 1 else "Bob"