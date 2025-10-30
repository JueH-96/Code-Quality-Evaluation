class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        
        max_obtainable = 0  # Maximum value we can currently obtain
        added_coins = 0
        i = 0  # Index for coins array
        
        while max_obtainable < target:
            # If we have a coin that can extend our current range
            if i < len(coins) and coins[i] <= max_obtainable + 1:
                max_obtainable += coins[i]
                i += 1
            else:
                # We need to add a coin to fill the gap
                # The optimal coin to add is (max_obtainable + 1)
                max_obtainable += (max_obtainable + 1)
                added_coins += 1
        
        return added_coins