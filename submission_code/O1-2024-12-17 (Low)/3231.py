class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # Sort the existing coins
        coins.sort()
        
        # current_reach tracks the maximum value we can sum to so far (from 1 to current_reach inclusive)
        current_reach = 0
        added_coins = 0
        
        i = 0
        n = len(coins)
        
        # While we haven't covered all values up to 'target'
        while current_reach < target:
            # If we still have coins left and the next coin is within the gap 
            # ( <= current_reach + 1 ), use it to extend current coverage
            if i < n and coins[i] <= current_reach + 1:
                current_reach += coins[i]
                i += 1
            else:
                # Otherwise, we need to add a coin of value (current_reach + 1)
                current_reach += (current_reach + 1)
                added_coins += 1
        
        return added_coins