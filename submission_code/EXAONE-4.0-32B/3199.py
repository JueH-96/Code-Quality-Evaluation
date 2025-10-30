class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def f(t):
            if t < 0:
                return 0
            return (t + 2) * (t + 1) // 2
        
        return f(n) - 3 * f(n - (limit + 1)) + 3 * f(n - 2 * (limit + 1)) - f(n - 3 * (limit + 1))