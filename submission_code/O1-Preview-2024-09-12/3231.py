from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        miss = 1
        count = 0
        i = 0
        n = len(coins)
        while miss <= target:
            if i < n and coins[i] <= miss:
                miss += coins[i]
                i +=1
            else:
                miss += miss
                count +=1
        return count