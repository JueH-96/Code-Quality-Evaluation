class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize a list to keep track of who has received the ball
        received = [0] * (n + 1)
        # Start at the 1^st friend
        current = 1
        # Start passing the ball
        for i in range(1, n + 1):
            # Pass the ball to the friend i * k steps away
            current = (current + i * k) % n
            # If the current friend has received the ball for the first time, mark them as received
            if current == 0:
                current = n
            received[current] = 1
        # Return the friends who have not received the ball
        return [i for i in range(1, n + 1) if received[i] == 0]