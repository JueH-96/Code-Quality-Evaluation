class Solution:
    def canAliceWin(self, n: int) -> bool:
        
        remaining_stones = n
        required_move_size = 10  # Alice's first move is always 10
        is_alice_turn = True

        # Simulate the game turn by turn
        while True:
            # Check if the current player can make a valid move.
            # A move is valid only if:
            # 1. The required_move_size is positive.
            # 2. There are enough stones remaining in the pile.
            if remaining_stones < required_move_size or required_move_size <= 0:
                # The current player cannot make a valid move, so they lose.
                # If it's Alice's turn and she can't move, Alice loses (return False).
                # If it's Bob's turn and he can't move, Bob loses, which means Alice wins (return True).
                return not is_alice_turn
            
            # The current player makes the move by removing stones
            remaining_stones -= required_move_size
            last_move_made = required_move_size # Store the size of the move just made

            # Prepare for the next turn:
            # The next player must remove 1 fewer stone than the current player just removed.
            required_move_size = last_move_made - 1
            # Switch the turn to the other player
            is_alice_turn = not is_alice_turn