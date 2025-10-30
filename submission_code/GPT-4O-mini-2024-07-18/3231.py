class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins = sorted(set(coins))  # Remove duplicates and sort the coins
        current_sum = 0
        added_coins = 0
        next_coin = 1
        
        for coin in coins:
            while next_coin <= target and next_coin <= current_sum + 1:
                current_sum += next_coin
                added_coins += 1
                next_coin *= 2
            
            current_sum += coin
            
            if current_sum >= target:
                break
        
        while next_coin <= target:
            current_sum += next_coin
            added_coins += 1
            next_coin *= 2
        
        return added_coins