class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        reachable = 0
        added_coins = 0
        for coin in coins:
            if coin > reachable + 1:
                while reachable < target and coin > reachable + 1:
                    reachable += reachable + 1
                    added_coins += 1
            reachable += coin
            if reachable >= target:
                break
        while reachable < target:
            reachable += reachable + 1
            added_coins += 1
        return added_coins