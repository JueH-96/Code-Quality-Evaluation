from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # Sort the coins to apply the greedy patching strategy
        coins.sort()
        
        patches = 0    # number of coins added
        reach = 0      # current maximum sum [1..reach] we can form
        i = 0
        n = len(coins)
        
        # While we cannot yet form all values up to target
        while reach < target:
            # If the next coin can extend our coverage
            if i < n and coins[i] <= reach + 1:
                reach += coins[i]
                i += 1
            else:
                # Patch by adding a coin of value reach+1
                patches += 1
                reach += (reach + 1)
                
        return patches