class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Alice must remove 10 stones on her first turn.
        # If n < 10, Alice cannot move and loses immediately.
        # Otherwise, we alternate removing 1 fewer stone each turn
        # until one player is unable to remove the required stones.
        
        stones_left = n
        current_player_is_alice = True
        stones_to_remove = 10  # Alice's first move requires removing 10 stones
        
        while True:
            # If there aren't enough stones to remove, the current player loses
            if stones_left < stones_to_remove:
                # If it's Alice's turn and she cannot move, Alice loses
                # Otherwise, Bob loses
                return not current_player_is_alice
            
            # Make the move
            stones_left -= stones_to_remove
            
            # Switch turns
            current_player_is_alice = not current_player_is_alice
            
            # The next move requires one fewer stone to be removed
            stones_to_remove -= 1
            
            # If stones_to_remove drops below 1, no further moves are possible
            # The next player to move would be unable to move, so the other player wins
            if stones_to_remove < 1:
                # The next player can't move, so the current player is the last to have made a valid move
                # That means the player who just made the move (which is the opposite of current_player_is_alice now) wins
                return not current_player_is_alice