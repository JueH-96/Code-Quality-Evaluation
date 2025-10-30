class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_max = 0
        count = 0
        i = 0
        n = len(coins)
        
        while current_max < target and i < n:
            if coins[i] > current_max + 1:
                current_max += (current_max + 1)
                count += 1
            else:
                current_max += coins[i]
                i += 1
                
        while current_max < target:
            current_max += (current_max + 1)
            count += 1
            
        return count