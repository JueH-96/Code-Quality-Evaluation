class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize a list to keep track of who has received the ball
        received_ball = [False] * n
        # Start with the first friend
        current_friend = 0
        # Initialize a counter to keep track of the number of passes
        pass_count = 0
        
        while True:
            # Calculate the next friend to receive the ball
            current_friend = (current_friend + pass_count * k) % n
            # If the current friend has already received the ball, break the loop
            if received_ball[current_friend]:
                break
            # Mark the current friend as having received the ball
            received_ball[current_friend] = True
            # Increment the pass count
            pass_count += 1
        
        # Find all friends who did not receive the ball
        losers = [i + 1 for i, received in enumerate(received_ball) if not received]
        return losers

# The List type needs to be imported from typing module
from typing import List