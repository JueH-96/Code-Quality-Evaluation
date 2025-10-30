class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added, i, max_sum = 0, 0, 0
        while max_sum < target:
            if i < len(coins) and coins[i] <= max_sum + 1:
                max_sum += coins[i]
                i += 1
            else:
                max_sum += max_sum + 1
                added += 1
        return added