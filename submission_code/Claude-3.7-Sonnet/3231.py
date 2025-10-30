class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()  # Sort coins to process them in order
        max_obtainable = 0  # Maximum value we can currently obtain
        added = 0  # Count of coins we've added
        idx = 0  # Index to track position in coins array
        
        while max_obtainable < target:
            # If we have a coin that can extend our current range
            if idx < len(coins) and coins[idx] <= max_obtainable + 1:
                max_obtainable += coins[idx]
                idx += 1
            # Otherwise, we need to add a new coin
            else:
                added += 1
                # Add a coin with value (max_obtainable + 1)
                max_obtainable += (max_obtainable + 1)
        
        return added