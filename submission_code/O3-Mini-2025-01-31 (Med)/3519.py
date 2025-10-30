from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # We'll maintain a dictionary for each player to count the colors.
        player_counts = [defaultdict(int) for _ in range(n)]
        
        # Process each pick.
        for player, color in pick:
            player_counts[player][color] += 1
        
        winners = 0
        # For each player, check if any color count is at least (player index + 1)
        for i in range(n):
            for count in player_counts[i].values():
                if count >= i + 1:
                    winners += 1
                    break  # Once win condition met, no need to check further.
        return winners

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    n1 = 4
    pick1 = [[0,0],[1,0],[1,0],[2,1],[2,1],[2,0]]
    print(sol.winningPlayerCount(n1, pick1))  # Expected output: 2

    # Example 2:
    n2 = 5
    pick2 = [[1,1],[1,2],[1,3],[1,4]]
    print(sol.winningPlayerCount(n2, pick2))  # Expected output: 0

    # Example 3:
    n3 = 5
    pick3 = [[1,1],[2,4],[2,4],[2,4]]
    print(sol.winningPlayerCount(n3, pick3))  # Expected output: 1