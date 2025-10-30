from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        obtainable = 0
        added_coins = 0
        i = 0
        
        while obtainable < target:
            if i < len(coins) and coins[i] <= obtainable + 1:
                obtainable += coins[i]
                i += 1
            else:
                obtainable += obtainable + 1
                added_coins += 1
        
        return added_coins