class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reachable = 0
        added_coins = 0
        i = 0
        n = len(coins)
        
        while max_reachable < target:
            if i < n and coins[i] <= max_reachable + 1:
                max_reachable += coins[i]
                i += 1
            else:
                added_coins += 1
                max_reachable += max_reachable + 1
        
        return added_coins