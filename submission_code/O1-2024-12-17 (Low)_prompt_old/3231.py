class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added_coins = 0
        reachable_sum = 0
        i = 0
        n = len(coins)
        
        # While we haven't yet covered sums up to target
        while reachable_sum < target:
            # If we still have coins left and the next coin is within the gap we can cover
            if i < n and coins[i] <= reachable_sum + 1:
                reachable_sum += coins[i]
                i += 1
            else:
                # We need to add a coin with value (reachable_sum + 1)
                added_coins += 1
                reachable_sum += (reachable_sum + 1)
        
        return added_coins