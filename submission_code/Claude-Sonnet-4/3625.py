class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones_to_remove = 10
        alice_turn = True
        
        while stones_to_remove <= n:
            n -= stones_to_remove
            stones_to_remove -= 1
            alice_turn = not alice_turn
        
        # If it's Alice's turn and she can't make a move, she loses
        # If it's Bob's turn and he can't make a move, Alice wins
        return not alice_turn