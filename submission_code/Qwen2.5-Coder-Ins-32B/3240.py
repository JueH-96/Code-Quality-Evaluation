class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(n, x):
            count = 0
            for i in range(1, n + 1):
                num = i
                pos = 1
                while num > 0:
                    if pos % x == 0 and num % 2 == 1:
                        count += 1
                    num //= 2
                    pos += 1
            return count

        low, high = 1, k + 1
        while low < high:
            mid = (low + high) // 2
            if count_set_bits(mid, x) <= k:
                low = mid + 1
            else:
                high = mid
        return low - 1