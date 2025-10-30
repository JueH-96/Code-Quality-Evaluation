class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a dictionary to store colors count for each player.
        # Using list of dictionaries since player ids range from 0 to n-1.
        # Each dictionary will map color -> count.
        player_counts = [{} for _ in range(n)]
        
        # Fill in the counts from the picks list.
        for player, color in pick:
            if color in player_counts[player]:
                player_counts[player][color] += 1
            else:
                player_counts[player][color] = 1
        
        # Count the number of winners.
        winners = 0
        for i in range(n):
            required = i + 1  # Player i wins if they have at least i+1 balls of the same color.
            # Check if player i meets the condition.
            for count in player_counts[i].values():
                if count >= required:
                    winners += 1
                    break  # Only count once per player.
        return winners