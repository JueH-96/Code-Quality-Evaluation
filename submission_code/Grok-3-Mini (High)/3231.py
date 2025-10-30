class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added = 0
        current_max = 0
        idx = 0
        while current_max < target:
            if idx < len(coins) and coins[idx] <= current_max + 1:
                current_max += coins[idx]
                idx += 1
            else:
                # Add a new coin of value current_max + 1
                added += 1
                new_coin_value = current_max + 1
                current_max += new_coin_value
        return added