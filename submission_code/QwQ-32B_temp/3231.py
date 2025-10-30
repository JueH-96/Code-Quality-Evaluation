class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_max = 0
        res = 0
        index = 0
        n = len(coins)
        
        while current_max < target:
            if index < n and coins[index] <= current_max + 1:
                current_max += coins[index]
                index += 1
            else:
                res += 1
                current_max += current_max + 1  # add the coin current_max + 1
        
        return res