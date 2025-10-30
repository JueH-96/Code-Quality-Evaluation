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
                added_coin = current_max + 1
                res += 1
                current_max += added_coin
        return res