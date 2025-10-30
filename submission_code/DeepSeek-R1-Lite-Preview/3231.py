class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reachable = 0
        added = 0
        i = 0
        n = len(coins)
        
        while max_reachable < target:
            if i < n and coins[i] <= max_reachable + 1:
                max_reachable += coins[i]
                i += 1
            else:
                # Need to add a coin with value max_reachable + 1
                added += 1
                max_reachable += max_reachable + 1
        return added