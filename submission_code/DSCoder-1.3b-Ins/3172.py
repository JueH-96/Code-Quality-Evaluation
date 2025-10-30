class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        def sum_not_divisible(n, m):
            return sum(i for i in range(1, n + 1) if i % m != 0)

        def sum_divisible(n, m):
            return sum(i for i in range(1, n + 1) if i % m == 0)

        return sum_not_divisible(n, m) - sum_divisible(n, m)