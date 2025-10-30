class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        def check(num):
            total = 0
            for i in range(1, num + 1):
                if i % x == 0:
                    total += count_set_bits(i)
            return total <= k

        left, right = 1, 10**15
        while left < right:
            mid = (left + right + 1) // 2
            if check(mid):
                left = mid
            else:
                right = mid - 1
        return left