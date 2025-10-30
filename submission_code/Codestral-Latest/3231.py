class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reachable = 0
        i = 0
        additional_coins = 0

        while max_reachable < target:
            if i < len(coins) and coins[i] <= max_reachable + 1:
                max_reachable += coins[i]
                i += 1
            else:
                max_reachable += max_reachable + 1
                additional_coins += 1

        return additional_coins