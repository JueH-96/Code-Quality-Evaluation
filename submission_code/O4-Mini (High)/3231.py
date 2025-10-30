from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # Sort existing coins
        coins.sort()
        miss = 1      # smallest sum we cannot form yet
        added = 0     # number of coins added
        i = 0
        n = len(coins)
        
        # While there is some value up to target we cannot form
        while miss <= target:
            # If the next existing coin can extend our reach, use it
            if i < n and coins[i] <= miss:
                miss += coins[i]
                i += 1
            else:
                # Otherwise, we "patch" by adding a coin of value = miss
                added += 1
                miss += miss
        
        return added