from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count(mid):
            cnt = 0
            for coin in coins:
                cnt += mid // coin
            return cnt

        l, r = 0, 2 * 10**9
        while l < r:
            mid = (l + r) // 2
            if count(mid) < k:
                l = mid + 1
            else:
                r = mid
        return l