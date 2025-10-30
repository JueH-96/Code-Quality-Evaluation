from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()  # Sort the coins to process in increasing order
        missing = 1  # The smallest value we cannot obtain yet
        added_coins = 0
        i = 0
        n = len(coins)
        
        # Loop until we've covered all values up to target
        while missing <= target:
            # If the next coin is within the current gap, use it to extend the reach.
            if i < n and coins[i] <= missing:
                missing += coins[i]
                i += 1
            else:
                # Otherwise, add a coin of value 'missing' to cover the current gap.
                added_coins += 1
                missing += missing  # Using the patch, we extend the range to [1, 2*missing - 1]
        return added_coins