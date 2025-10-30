class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict

        # Dictionary to store counts of (player -> color -> occurrences)
        freq = defaultdict(lambda: defaultdict(int))
        
        for x, y in pick:
            freq[x][y] += 1
        
        winners = 0
        # Check each player for winning condition
        for player in range(n):
            required_count = player + 1
            # If any color count >= required_count, the player wins
            if any(count >= required_count for count in freq[player].values()):
                winners += 1
        
        return winners