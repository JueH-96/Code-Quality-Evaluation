class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False  # Alice cannot make the first move
        
        pile = n
        stones_to_remove = 10
        alice_turn = True  # Alice starts first
        
        while True:
            if pile < stones_to_remove:
                if alice_turn:
                    return False  # Alice cannot move, she loses
                else:
                    return True   # Bob cannot move, Alice wins
            # Remove the stones
            pile -= stones_to_remove
            # Decrease the stones_to_remove by one for the next turn
            stones_to_remove -= 1
            # Switch turns
            alice_turn = not alice_turn