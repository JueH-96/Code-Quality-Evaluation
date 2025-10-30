class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count_less_equal(x):
            # Count unique multiples of coins <= x
            unique_multiples = set()
            for coin in coins:
                multiple = coin
                while multiple <= x:
                    unique_multiples.add(multiple)
                    multiple += coin
                    # Early termination if we've already found k or more
                    if len(unique_multiples) >= k:
                        return len(unique_multiples)
            return len(unique_multiples)
        
        # Binary search range
        left = 1
        right = min(coins) * k  # Upper bound: k times the smallest coin
        
        while left < right:
            mid = (left + right) // 2
            if count_less_equal(mid) < k:
                left = mid + 1
            else:
                right = mid
        
        return left