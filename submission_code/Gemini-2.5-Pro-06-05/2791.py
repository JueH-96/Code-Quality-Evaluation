class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # Use a boolean array to track which friends have received the ball.
        # Index i corresponds to friend i+1.
        has_received = [False] * n
        
        # Friend 1 (index 0) starts with the ball.
        current_player_idx = 0
        has_received[current_player_idx] = True
        
        turn = 1
        
        # Simulate the game turn by turn.
        while True:
            # Calculate the pass distance for the current turn.
            pass_distance = turn * k
            
            # Calculate the next player's index in the circular arrangement.
            next_player_idx = (current_player_idx + pass_distance) % n
            
            # The game ends if the next player has already received the ball.
            if has_received[next_player_idx]:
                break
            
            # Otherwise, update the state for the next turn.
            has_received[next_player_idx] = True
            current_player_idx = next_player_idx
            turn += 1
            
        # After the game, identify the losers.
        # Losers are friends whose corresponding entry in has_received is False.
        losers = []
        for i in range(n):
            if not has_received[i]:
                # Friend numbers are 1-based, so add 1 to the index.
                losers.append(i + 1)
        
        # The losers are collected in ascending order because we iterate from 0 to n-1.
        return losers