class Solution:
    def canAliceWin(self, n: int) -> bool:
        current_stones = n
        
        # The sequence of stones players must remove is 10, 9, 8, ..., 1.
        # Alice makes the first move (10), then Bob (9), then Alice (8), and so on.
        # Alice's turns require removing 10, 8, 6, 4, 2 stones.
        # Bob's turns require removing 9, 7, 5, 3, 1 stones.
        
        for stones_to_remove in range(10, 0, -1):
            # Check if the current player can make the required move (remove stones_to_remove).
            if current_stones < stones_to_remove:
                # The current player cannot make the move because there are not enough stones.
                # The current player loses the game.
                
                # Determine whose turn it was based on the value of stones_to_remove.
                # The required number of stones (k) tells us whose turn it is in the sequence:
                # If k is even (10, 8, 6, 4, or 2), it was Alice's turn.
                # If k is odd (9, 7, 5, 3, or 1), it was Bob's turn.
                
                if stones_to_remove % 2 == 0:
                    # It was Alice's turn. Alice cannot make the move, so Alice loses.
                    return False
                else:
                    # It was Bob's turn. Bob cannot make the move, so Bob loses.
                    # Since Bob loses, Alice wins.
                    return True
            else:
                # The current player can make the move. Remove the required number of stones.
                current_stones -= stones_to_remove
                # The turn automatically alternates for the next iteration of the loop
                # as the loop progresses through the sequence 10, 9, ..., 1.

        # If the loop finishes, it means all moves from 10 down to 1 were successfully made.
        # The last move (removing 1 stone) was made by Bob (since 1 is an odd value in the sequence).
        # After Bob removes 1 stone, it becomes Alice's turn. The next required move in the sequence
        # would be 1 - 1 = 0. A player cannot remove 0 stones.
        # Alice cannot make a move (remove 0 stones), so she cannot make a move according to the rules.
        # Alice loses the game. Bob wins.
        # Note: Given the constraint n <= 50, the sum of stones to remove (1..10) is 55.
        # For n <= 50, the game will always end before the loop finishes, as current_stones will
        # become less than stones_to_remove at some point. However, this return handles the
        # logical outcome if the sequence *could* be completed (e.g., if n >= 55).
        return False # Alice loses if the entire sequence of moves is possible