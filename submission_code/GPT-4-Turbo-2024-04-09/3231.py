class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_max_reachable = 0
        added_coins = 0
        i = 0
        
        while current_max_reachable < target:
            if i < len(coins) and coins[i] <= current_max_reachable + 1:
                current_max_reachable += coins[i]
                i += 1
            else:
                added_coins += 1
                current_max_reachable += current_max_reachable + 1
        
        return added_coins