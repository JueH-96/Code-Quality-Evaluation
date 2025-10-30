class Solution:
  def canAliceWin(self, n: int) -> bool:
    """
    Simulates the stone game to determine if Alice wins.

    Args:
      n: The initial number of stones.

    Returns:
      True if Alice wins, False otherwise.
    """
    
    stones_left = n
    
    # Alice's first turn
    # Alice must remove 10 stones.
    alice_first_move_amount = 10
    
    if stones_left < alice_first_move_amount:
      # Alice cannot make her first move if n < 10.
      # Alice loses.
      return False 
    
    stones_left -= alice_first_move_amount
    # `stones_removed_by_last_player` will store the number of stones
    # removed by the player whose turn just ended. For Bob's upcoming turn,
    # Alice was the last player, and she removed `alice_first_move_amount` stones.
    stones_removed_by_last_player = alice_first_move_amount

    # The game continues in a loop, starting with Bob's turn.
    # Each iteration of the while loop covers one turn for Bob and one turn for Alice.
    while True:
      # Bob's turn
      # Bob must remove 1 fewer stone than Alice's last move.
      bob_attempt_remove = stones_removed_by_last_player - 1
      
      if bob_attempt_remove <= 0:
        # Bob is asked to remove 0 or a negative number of stones.
        # A move must involve removing a positive number of stones.
        # Thus, Bob cannot make a valid move. Bob loses. Alice wins.
        return True 
      
      if stones_left < bob_attempt_remove:
        # Bob does not have enough stones to make his required move.
        # Bob loses. Alice wins.
        return True
      
      # Bob successfully makes his move.
      stones_left -= bob_attempt_remove
      # For Alice's upcoming turn, Bob was the last player.
      stones_removed_by_last_player = bob_attempt_remove 

      # Alice's turn (this is for Alice's second, third, etc. turns)
      # Alice must remove 1 fewer stone than Bob's last move.
      alice_attempt_remove = stones_removed_by_last_player - 1
      
      if alice_attempt_remove <= 0:
        # Alice is asked to remove 0 or a negative number of stones.
        # Alice cannot make a valid move. Alice loses.
        return False
            
      if stones_left < alice_attempt_remove:
        # Alice does not have enough stones to make her required move.
        # Alice loses.
        return False
      
      # Alice successfully makes her move.
      stones_left -= alice_attempt_remove
      # For Bob's upcoming turn, Alice was the last player.
      stones_removed_by_last_player = alice_attempt_remove