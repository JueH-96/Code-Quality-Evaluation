class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        def f(x, y):
            if x < y:
                return y - x
            if x == y:
                return 0
            if x % 11 == 0 and x // 11 >= y:
                return 1 + f(x // 11, y)
            if x % 5 == 0 and x // 5 >= y:
                return 1 + f(x // 5, y)
            return 1 + min(f(x - 1, y), f(x + 1, y))
        return f(x, y)