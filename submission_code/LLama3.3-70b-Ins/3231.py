from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reachable = 0
        count = 0
        
        for coin in coins:
            if max_reachable + 1 < coin:
                count += 1
                max_reachable = 2 * max_reachable + 1
            max_reachable += coin
        
        if max_reachable < target:
            count += (target - max_reachable + max_reachable) // (max_reachable + 1)
        
        return count