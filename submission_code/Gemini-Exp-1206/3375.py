class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        left, right = 1, 25 * k
        while left < right:
            mid = (left + right) // 2
            count = 0
            for coin in coins:
                count += mid // coin
            if count < k:
                left = mid + 1
            else:
                right = mid
        return left