from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Create a list to store the number of times each friend receives the ball.
        # Indices 0 to n-1 correspond to friends 1 to n.
        # Initialize all counts to 0.
        received_count = [0] * n

        # The ball starts with friend 1 (index 0).
        current_idx = 0

        # According to the rules, friend 1 receives the ball initially.
        # We mark their count as 1. The game ends when someone receives it for the *second* time.
        received_count[current_idx] = 1

        # The game proceeds in turns, starting from turn 1.
        turn = 1

        # Simulate the game turn by turn until a friend receives the ball twice.
        while True:
            # Calculate the number of steps for the current turn.
            # On turn i, the pass is i * k steps away.
            steps = turn * k

            # Calculate the index of the next friend to receive the ball.
            # Start from the current friend's index, add the steps, and use modulo n
            # to handle the circular arrangement.
            next_idx = (current_idx + steps) % n

            # Increment the received count for the friend at next_idx.
            received_count[next_idx] += 1

            # Check if the game ends.
            # The game ends when the friend at next_idx has now received the ball 2 times.
            if received_count[next_idx] == 2:
                break # Game is over, exit the simulation loop

            # If the game doesn't end, update the current ball holder's index
            # and move to the next turn.
            current_idx = next_idx
            turn += 1

        # After the game ends, identify the losers.
        # Losers are friends who never received the ball, meaning their count is still 0.
        losers = []
        # Iterate through all possible friend indices (0 to n-1).
        for i in range(n):
            # If the received count for this index is 0, the corresponding friend is a loser.
            if received_count[i] == 0:
                # Convert the 0-based index back to 1-based friend number (i + 1).
                losers.append(i + 1)

        # The problem asks for the losers in ascending order.
        # Since we iterate through indices i from 0 to n-1, the friend numbers i+1
        # are added to the 'losers' list in ascending order naturally.
        # No explicit sorting is required.

        return losers