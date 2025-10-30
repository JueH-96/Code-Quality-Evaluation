from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # Sort the existing coins so we can consume them in ascending order
        coins.sort()
        
        added = 0        # number of coins we add
        reach = 0        # with current set of coins, we can form every value in [1, reach]
        i = 0            # index for traversing the sorted coins list
        
        # Continue until every integer up to `target` is obtainable
        while reach < target:
            # If the next existing coin extends the coverage (<= reach + 1), use it
            if i < len(coins) and coins[i] <= reach + 1:
                reach += coins[i]
                i += 1
            else:
                # Otherwise, add a new coin of value reach + 1 (the minimal value needed)
                reach += reach + 1
                added += 1
                
        return added