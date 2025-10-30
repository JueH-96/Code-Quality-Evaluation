class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Initialize a list to track how many times each friend receives the ball
        received_count = [0] * n
        
        # Start the game with the first friend
        current_friend = 0
        step_multiplier = 1
        
        # Continue the game until a friend receives the ball for the second time
        while received_count[current_friend] < 2:
            # Mark that the current friend has received the ball
            received_count[current_friend] += 1
            
            # Calculate the next friend to receive the ball
            current_friend = (current_friend + step_multiplier * k) % n
            
            # Increment the step multiplier for the next pass
            step_multiplier += 1
        
        # Collect all friends who never received the ball
        losers = [i + 1 for i in range(n) if received_count[i] == 0]
        
        return losers