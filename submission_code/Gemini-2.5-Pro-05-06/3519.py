from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # player_ball_counts[player_id][color_id] will store the count of balls
        # of color_id picked by player_id.
        # Max color_id is 10 (from constraints: 0 <= y_i <= 10).
        # So, we need 11 columns for colors 0 through 10.
        num_distinct_colors = 11  # For colors 0, 1, ..., 10
        
        # Initialize a 2D list (n rows for players, num_distinct_colors columns for colors) 
        # with all zeros.
        player_ball_counts = [[0] * num_distinct_colors for _ in range(n)]

        # Process the pick array to populate the counts.
        # Each item in pick is [x_i, y_i], where x_i is player_id and y_i is color_id.
        for p_id, c_id in pick:
            player_ball_counts[p_id][c_id] += 1

        winning_players_count = 0
        
        # Iterate through each player to check if they win.
        # Player IDs are 0, 1, ..., n-1.
        for player_id in range(n):
            # Determine the winning condition for the current player_id.
            # Player 'i' (here, player_id) wins if they pick strictly more than 'i' balls
            # of the same color. This means they need at least 'i' + 1 balls of that color.
            required_balls = player_id + 1
            
            is_winner = False
            # Check the ball counts for the current player for each color.
            for color_index in range(num_distinct_colors):
                if player_ball_counts[player_id][color_index] >= required_balls:
                    is_winner = True
                    break  # Condition met for this player, no need to check other colors.
            
            if is_winner:
                winning_players_count += 1
                
        return winning_players_count