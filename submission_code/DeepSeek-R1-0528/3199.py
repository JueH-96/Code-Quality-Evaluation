class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def f(k):
            if k < 0:
                return 0
            return (k + 2) * (k + 1) // 2
        
        total = f(n)
        a = f(n - (limit + 1))
        b = f(n - 2 * (limit + 1))
        c = f(n - 3 * (limit + 1))
        
        return total - 3 * a + 3 * b - c