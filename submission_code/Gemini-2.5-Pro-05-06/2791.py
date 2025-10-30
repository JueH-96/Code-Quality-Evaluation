from typing import List

class Solution:
  def circularGameLosers(self, n: int, k: int) -> List[int]:
    # has_received_ball[i] is true if friend (i+1) (1-indexed) has received the ball.
    # We use 0-indexed friends internally for easier modulo arithmetic:
    # Friend 1 (1-indexed) -> index 0
    # Friend 2 (1-indexed) -> index 1
    # ...
    # Friend n (1-indexed) -> index n-1
    
    # Initialize a boolean list to track which friends have received the ball.
    # `has_received_ball[j]` is True if friend with 0-index `j` received the ball.
    has_received_ball = [False] * n
    
    # Friend 1 (0-indexed as 0) initially has the ball.
    current_friend_0_indexed = 0
    has_received_ball[current_friend_0_indexed] = True
    
    # The variable `pass_turn_multiplier` corresponds to 'i' in the problem description:
    # "on the i^th turn, the friend holding the ball should pass it to the friend who is i * k steps away".
    # The first pass is effectively the 1st turn for calculating steps. So this multiplier starts at 1.
    pass_turn_multiplier = 1
    
    # Simulate the game.
    while True:
        # Calculate the number of steps for the current pass.
        steps_to_take = pass_turn_multiplier * k
        
        # Determine the recipient of the pass (0-indexed).
        # Moving `steps_to_take` clockwise from `current_friend_0_indexed`.
        next_friend_0_indexed = (current_friend_0_indexed + steps_to_take) % n
        
        # Check if this recipient has already received the ball.
        if has_received_ball[next_friend_0_indexed]:
            # If True, this friend receives the ball for the second time. The game ends.
            break
        else:
            # This friend receives the ball for the first time.
            has_received_ball[next_friend_0_indexed] = True  # Mark as having received the ball.
            # This friend will make the next pass.
            current_friend_0_indexed = next_friend_0_indexed
            # Increment the multiplier for the next pass.
            pass_turn_multiplier += 1
            
    # The game has concluded. Now, identify the losers.
    # Losers are friends who never received the ball.
    # Their numbers should be 1-indexed and in ascending order.
    losers_1_indexed = []
    for i in range(n):  # Iterate through 0-indexed friend numbers (0 to n-1).
        if not has_received_ball[i]:
            # If friend with 0-index `i` did not receive the ball,
            # then friend `i+1` (1-indexed) is a loser.
            losers_1_indexed.append(i + 1)
            
    # The `losers_1_indexed` list is already sorted in ascending order because
    # we iterated `i` from 0 up to `n-1`.
    return losers_1_indexed