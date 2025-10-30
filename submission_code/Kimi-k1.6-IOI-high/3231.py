class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reachable = 0
        res = 0
        i = 0
        n = len(coins)
        
        while max_reachable < target:
            if i < n and coins[i] <= max_reachable + 1:
                max_reachable += coins[i]
                i += 1
            else:
                max_reachable += max_reachable + 1
                res += 1
        
        return res