class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        reach = 0
        added_coins = 0
        i = 0
        
        # While we haven't covered up to 'target'
        while reach < target:
            # If there's a coin available that can extend coverage
            # (i.e., it is at most reach+1), use it.
            if i < len(coins) and coins[i] <= reach + 1:
                reach += coins[i]
                i += 1
            else:
                # Otherwise, add a new coin of value (reach+1)
                added_coins += 1
                reach += (reach + 1)
        
        return added_coins