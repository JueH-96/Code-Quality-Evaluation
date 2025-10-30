class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        if s1 != s2:
            return -1
        min_cost = x * (n // 2)
        if n % 2 == 1:
            min_cost += x
        return min_cost