class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count_numbers_less_than(x):
            count = 0
            for coin in coins:
                count += (x // coin)
            return count
        
        left, right = 1, k * min(coins)
        
        while left < right:
            mid = (left + right) // 2
            
            if count_numbers_less_than(mid) >= k:
                right = mid
            else:
                left = mid + 1
                
        return left