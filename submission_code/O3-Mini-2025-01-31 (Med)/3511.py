class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # To reach exactly 115, the only valid combination is:
        # 1 coin of 75 and 4 coins of 10 (since 75 + 4 * 10 = 115).
        # Therefore, in each turn the player must pick 1 coin of 75 and 4 coins of 10.
        # The maximum number of complete turns possible is limited by:
        #   - The available number of 75 coins: x turns.
        #   - The available number of 10 coins divided by 4: y // 4 turns.
        # So, turns = min(x, y // 4)
        # If turns is odd, Alice (the starting player) wins, 
        # otherwise Bob wins.
        
        turns = min(x, y // 4)
        return "Alice" if turns % 2 == 1 else "Bob"