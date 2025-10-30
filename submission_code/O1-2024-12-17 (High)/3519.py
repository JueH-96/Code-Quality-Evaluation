class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict

        # Frequency of colors picked by each player
        freq = [defaultdict(int) for _ in range(n)]
        
        # Populate the frequency data
        for player, color in pick:
            freq[player][color] += 1
        
        # Count how many players meet the winning criteria
        winners = 0
        for i in range(n):
            # Player i needs at least (i+1) of the same color
            if any(count >= (i + 1) for count in freq[i].values()):
                winners += 1
        
        return winners