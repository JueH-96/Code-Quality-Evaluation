from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize a boolean array to keep track of who received the ball.
        # We use 1-based indexing for friends (from 1 to n), so the array size is n+1.
        # received[i] will be True if friend i has received the ball.
        received = [False] * (n + 1)
        
        # Friend 1 starts with the ball.
        current_holder = 1
        received[current_holder] = True
        
        # The turn_multiplier determines how many steps (i * k) the ball is passed.
        # It starts from 1 for the 1st turn (1*k steps).
        turn_multiplier = 1
        
        # Simulate the game turn by turn.
        while True:
            # Calculate the number of steps for the current turn.
            steps = turn_multiplier * k
            
            # Calculate the next friend to receive the ball using 1-based circular indexing.
            # The formula (current_position + steps - 1) % total_friends + 1 correctly
            # handles wrapping around the circle for 1-indexed positions.
            # For example, if n=5, current_holder=1, steps=2:
            # (1 + 2 - 1) % 5 + 1 = 2 % 5 + 1 = 2 + 1 = 3.
            # If n=5, current_holder=3, steps=4:
            # (3 + 4 - 1) % 5 + 1 = 6 % 5 + 1 = 1 + 1 = 2.
            next_receiver = (current_holder + steps - 1) % n + 1
            
            # Check if this friend has already received the ball.
            if received[next_receiver]:
                # If true, the game ends because a friend received the ball for the second time.
                break
            else:
                # If false, this friend is receiving the ball for the first time.
                # Mark them as received.
                received[next_receiver] = True
                # Update the current holder to the friend who just received the ball.
                current_holder = next_receiver
                # Increment the turn multiplier for the next iteration.
                turn_multiplier += 1
                
        # After the game ends, identify the friends who never received the ball.
        losers = []
        # Iterate through friends from 1 to n.
        for i in range(1, n + 1):
            # If a friend's entry in the 'received' array is False, they are a loser.
            if not received[i]:
                losers.append(i)
                
        # The 'losers' list will naturally be in ascending order because we iterated from 1 to n.
        return losers