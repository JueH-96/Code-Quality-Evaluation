class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added = 0
        current_sum = 0
        coin_index = 0
        
        for value in range(1, target + 1):
            if coin_index < len(coins) and coins[coin_index] <= value:
                current_sum += coins[coin_index]
                coin_index += 1
            elif current_sum < value:
                added += 1
                current_sum += value
            
            if current_sum >= target:
                break
        
        return added