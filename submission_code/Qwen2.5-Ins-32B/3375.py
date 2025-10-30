from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def enough(x):
            count = 0
            for coin in coins:
                count += x // coin
            return count >= k
        
        coins.sort()
        left, right = 1, k * coins[0]
        while left < right:
            mid = (left + right) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left