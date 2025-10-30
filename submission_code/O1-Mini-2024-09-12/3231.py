class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added = 0
        i = 0
        max_reachable = 0
        while max_reachable < target:
            if i < len(coins) and coins[i] <= max_reachable + 1:
                max_reachable += coins[i]
                i += 1
            else:
                # Add the smallest missing coin
                added += 1
                max_reachable += max_reachable + 1
        return added