class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Alice can only start the game if there are at least 10 stones
        if n < 10:
            return False
        
        # After Alice removes 10 stones, the remaining stones
        remaining_stones = n - 10
        
        # Bob will then try to remove stones, starting with 9
        # We need to check if Alice can win from this point
        # The game alternates between Alice and Bob, reducing the number of stones removed by 1 each turn
        
        # The number of stones Bob can remove on his first turn
        bob_turn = 9
        
        # While there are stones left
        while remaining_stones > 0:
            # Bob's turn
            if remaining_stones >= bob_turn:
                remaining_stones -= bob_turn
            else:
                # Bob cannot make a valid move, Alice wins
                return True
            
            # Alice's turn, she removes one less stone than Bob
            alice_turn = bob_turn - 1
            
            if remaining_stones >= alice_turn:
                remaining_stones -= alice_turn
            else:
                # Alice cannot make a valid move, Bob wins
                return False
            
            # Prepare for the next round
            bob_turn -= 1
        
        # If we exit the loop, it means Bob has made the last valid move
        return False