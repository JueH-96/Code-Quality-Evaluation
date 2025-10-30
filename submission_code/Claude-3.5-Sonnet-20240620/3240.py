class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(n):
            count = 0
            for i in range(x, 64, x):
                count += (n >> (i - 1)) & 1
            return count

        def price_sum(n):
            total = 0
            for i in range(1, 64):
                full_groups = n >> i
                total += full_groups * (1 << (i - 1)) // x
                remaining = n & ((1 << i) - 1)
                total += max(0, remaining - (1 << (i - 1)) + 1) // x
            return total

        left, right = 1, 10**18
        while left < right:
            mid = (left + right + 1) // 2
            if price_sum(mid) <= k:
                left = mid
            else:
                right = mid - 1
        return left