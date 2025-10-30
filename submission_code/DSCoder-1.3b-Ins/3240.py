class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(n: int) -> int:
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        def check(n: int) -> bool:
            total = sum(count_set_bits(i) for i in range(1, n + 1))
            return total <= k

        left, right = 1, 10 ** 10
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid
        return left - 1