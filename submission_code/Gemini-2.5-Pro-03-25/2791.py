from typing import List # Required for type hinting List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        """
        Simulates the circular game described in the problem statement to find 
        the friends who never receive the ball (the losers).

        The game involves n friends in a circle numbered 1 to n. 
        Friend 1 starts with the ball.
        On turn `i` (starting from turn 1), the friend holding the ball passes it 
        to the friend who is `i * k` steps away in the clockwise direction.
        The game ends when a friend receives the ball for the second time.

        Args:
            n: The number of friends participating in the game. Friends are numbered 1 to n.
            k: An integer determining the step size multiplier for passing the ball.

        Returns:
            A list containing the numbers of the friends who are losers 
            (i.e., never received the ball during the game), sorted in ascending order.
        """

        # Use a set to keep track of friends who have received the ball.
        # Sets provide efficient membership testing (checking if a friend already received the ball).
        # Store friend numbers (which are 1-based).
        # Friend 1 starts with the ball, so they are the first to receive it.
        received_ball = {1}

        # Initialize the friend currently holding the ball. Starts with friend 1.
        current_friend = 1

        # Initialize the turn counter, starting from turn 1.
        turn = 1

        # Simulate the game turn by turn using a loop.
        # The loop continues until the game's end condition is met.
        while True:
            # Calculate the number of steps to pass the ball for the current turn.
            # On turn 'i', the step count is 'i * k'.
            steps = turn * k

            # Calculate the recipient of the ball for this turn.
            # The friends are in a circle, so we use modulo arithmetic.
            # We convert the 1-based 'current_friend' number to a 0-based index (current_friend - 1)
            # for the modulo calculation with 'n'.
            # (current_friend - 1 + steps) gives the target position relative to the start (friend 1 is at index 0).
            # The result of the modulo operation '% n' gives the 0-based index of the recipient.
            # We add 1 to convert the 0-based index back to a 1-based friend number.
            next_friend = ((current_friend - 1 + steps) % n) + 1

            # Check if the calculated 'next_friend' has already received the ball previously.
            if next_friend in received_ball:
                # If 'next_friend' is already in the 'received_ball' set, it means this friend
                # is receiving the ball for the second time. The game ends.
                break
            else:
                # If this is the first time 'next_friend' receives the ball:
                # 1. Add 'next_friend' to the set of friends who have received the ball.
                received_ball.add(next_friend)
                
                # 2. Update 'current_friend' to be 'next_friend' for the subsequent turn.
                current_friend = next_friend
                
                # 3. Increment the turn counter for the next iteration.
                turn += 1

        # After the game simulation loop terminates, we need to identify the losers.
        # Losers are the friends numbered from 1 to n who are *not* in the 'received_ball' set.
        losers = []
        for i in range(1, n + 1):
            # Iterate through all possible friend numbers (1 to n).
            if i not in received_ball:
                # If friend 'i' is not found in the set of those who received the ball,
                # they are a loser. Add them to the 'losers' list.
                losers.append(i)

        # The problem requires the list of losers to be sorted in ascending order.
        # Since we iterate from 1 to n when building the 'losers' list,
        # the list will naturally be sorted in ascending order.
        return losers