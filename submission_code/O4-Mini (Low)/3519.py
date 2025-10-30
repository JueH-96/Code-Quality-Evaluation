from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # counts[i][c] = number of balls of color c picked by player i
        counts = [defaultdict(int) for _ in range(n)]
        
        # Tally up the picks
        for player, color in pick:
            counts[player][color] += 1
        
        winners = 0
        # Check each player i: needs at least (i+1) balls of the same color
        for i in range(n):
            needed = i + 1
            # If any color count >= needed, player i wins
            for ccount in counts[i].values():
                if ccount >= needed:
                    winners += 1
                    break
        
        return winners

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.winningPlayerCount(4, [[0,0],[1,0],[1,0],[2,1],[2,1],[2,0]]))  # 2
    print(sol.winningPlayerCount(5, [[1,1],[1,2],[1,3],[1,4]]))            # 0
    print(sol.winningPlayerCount(5, [[1,1],[2,4],[2,4],[2,4]]))            # 1