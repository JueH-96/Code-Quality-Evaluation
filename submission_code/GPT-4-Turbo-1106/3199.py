class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def count_ways(n, limit, children):
            if children == 1:
                return 1 if 0 <= n <= limit else 0
            ways = 0
            for i in range(min(n, limit) + 1):
                ways += count_ways(n - i, limit, children - 1)
            return ways
        
        return count_ways(n, limit, 3)