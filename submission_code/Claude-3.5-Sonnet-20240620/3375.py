class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count_smaller_or_equal(x):
            count = 0
            for coin in coins:
                count += x // coin
            return count

        left, right = 1, min(coins) * k
        while left < right:
            mid = (left + right) // 2
            if count_smaller_or_equal(mid) < k:
                left = mid + 1
            else:
                right = mid
        return left