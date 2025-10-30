class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_max = 0
        i = 0
        added = 0
        
        while current_max < target:
            if i < len(coins) and coins[i] <= current_max + 1:
                current_max += coins[i]
                i += 1
            else:
                current_max += (current_max + 1)
                added += 1
        
        return added