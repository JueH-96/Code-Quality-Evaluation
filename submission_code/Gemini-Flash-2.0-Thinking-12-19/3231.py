class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        reachable_sum = 0
        added_coins = 0
        coin_idx = 0
        while reachable_sum < target:
            if coin_idx < len(coins) and coins[coin_idx] <= reachable_sum + 1:
                reachable_sum += coins[coin_idx]
                coin_idx += 1
            else:
                added_coins += 1
                reachable_sum += (reachable_sum + 1)
        return added_coins