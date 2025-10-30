class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        reachable_sum = 0
        added_coins_count = 0
        coin_index = 0
        while reachable_sum < target:
            if coin_index < len(coins) and coins[coin_index] <= reachable_sum + 1:
                reachable_sum += coins[coin_index]
                coin_index += 1
            else:
                added_coins_count += 1
                reachable_sum += (reachable_sum + 1)
        return added_coins_count