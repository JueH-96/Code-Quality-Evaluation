class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_sum = 0
        added_coins = 0
        i = 0
        
        while current_sum < target:
            if i < len(coins) and coins[i] <= current_sum + 1:
                current_sum += coins[i]
                i += 1
            else:
                current_sum += current_sum + 1
                added_coins += 1
        
        return added_coins