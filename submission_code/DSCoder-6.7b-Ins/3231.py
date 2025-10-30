class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added_coins = 0
        for i in range(len(coins)):
            if coins[i] <= target:
                target += coins[i]
                added_coins += 1
        return added_coins