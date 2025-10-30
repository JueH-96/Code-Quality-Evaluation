from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize a list to keep track of who has received the ball
        received = [0] * n
        # Start with the first friend
        received[0] = 1
        # The first friend starts with the ball
        current = 0
        # The number of steps starts at 1 and increments by 1 each turn
        steps = 1

        while True:
            # Calculate the next friend to receive the ball
            current = (current + steps * k) % n
            # Mark this friend as having received the ball
            received[current] += 1
            # If this friend has received the ball before, the game ends
            if received[current] > 1:
                break
            # Increment the steps for the next turn
            steps += 1

        # The losers are the friends who never received the ball
        losers = [i + 1 for i in range(n) if received[i] == 0]
        return losers