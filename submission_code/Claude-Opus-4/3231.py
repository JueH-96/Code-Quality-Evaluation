class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reach = 0
        added = 0
        i = 0
        
        while max_reach < target:
            # If we have a coin that can extend our range
            if i < len(coins) and coins[i] <= max_reach + 1:
                max_reach += coins[i]
                i += 1
            else:
                # We need to add a coin with value max_reach + 1
                max_reach += max_reach + 1
                added += 1
        
        return added