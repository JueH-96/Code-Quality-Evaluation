class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count(x):
            return sum(x // c for c in coins)
        
        left, right = 1, k * max(coins)
        while left < right:
            mid = (left + right) // 2
            if count(mid) < k:
                left = mid + 1
            else:
                right = mid
        return left