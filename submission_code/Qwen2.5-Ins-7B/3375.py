from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        coins.sort()
        n = len(coins)
        left, right = 0, 2 * 10**9
        
        while left < right:
            mid = (left + right) // 2
            count = sum(mid // coin for coin in coins)
            if count < k:
                left = mid + 1
            else:
                right = mid
        return left