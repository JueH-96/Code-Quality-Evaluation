class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins = sorted(coins)
        current_max = 0
        added = 0
        
        for coin in coins:
            if coin > current_max + 1:
                added += 1
                current_max += (current_max + 1)
            current_max += coin
        
        while current_max < target:
            added += 1
            current_max += (current_max + 1)
        
        return added